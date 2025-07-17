import os
import json
import time
import subprocess
import logging
import re
import configparser
import threading
from urllib.parse import urljoin, quote, urlparse
from functools import partial

import requests
import schedule

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("错误: watchdog 库未安装。请运行 'pip install watchdog' 来安装它。")
    exit(1)

# --- 全局变量 ---
ALIST_BASE_URL, ALIST_TOKEN = "", None
FFMPEG_PATH, THUMBNAIL_TIME = "ffmpeg", "00:00:10"
logger = logging.getLogger(__name__)
stop_scheduler_thread = threading.Event()


# <editor-fold desc="--- 核心处理函数 ---">
def sanitize_filename(filename):
    """清理文件名中的非法字符。"""
    sanitized = re.sub(r'[\\/*?:"<>|]', " ", filename)
    return re.sub(r"\s+", " ", sanitized).strip()


def get_strm_and_thumb_paths(relative_video_path, task_config):
    """根据相对路径和任务配置，计算出对应的.strm和缩略图文件的绝对路径。"""
    local_strm_base_dir = task_config["local_strm_dir"]
    local_thumb_base_dir = task_config.get("local_thumb_dir")

    # 检查是否为该任务启用了缩略图生成
    generate_thumbs_enabled = task_config.get("generate_thumbnails", False)

    sanitized_relative_path_parts = [
        sanitize_filename(part)
        for part in relative_video_path.replace("\\", "/").split("/")
    ]
    clean_relative_path = os.path.join(*sanitized_relative_path_parts)
    base_name, _ = os.path.splitext(clean_relative_path)

    strm_path = os.path.join(local_strm_base_dir, base_name + ".strm")

    # 仅在启用缩略图且配置了目录时才生成缩略图路径
    thumb_path = None
    if generate_thumbs_enabled and local_thumb_base_dir:
        thumb_format = task_config.get("thumbnail_format", "jpg")
        thumbnail_suffix = task_config.get("thumbnail_suffix", "").strip()

        final_base_name = base_name
        if thumbnail_suffix:
            final_base_name = f"{base_name}-{thumbnail_suffix}"

        thumb_path = os.path.join(
            local_thumb_base_dir, final_base_name + "." + thumb_format
        )

    return strm_path, thumb_path


def generate_thumbnail(video_source, thumb_path, task_config):
    """为单个视频源（URL或本地路径）生成缩略图。"""
    thumb_dir = os.path.dirname(thumb_path)
    os.makedirs(thumb_dir, exist_ok=True)
    if os.path.exists(thumb_path):
        logger.debug(f"缩略图已存在，跳过: {thumb_path}")
        return True

    thumbnail_size = task_config.get("thumbnail_size", "").strip()

    command = [FFMPEG_PATH]
    is_url = isinstance(video_source, str) and video_source.lower().startswith(
        ("http://", "https://")
    )
    if (
        is_url
        and ALIST_BASE_URL
        and video_source.startswith(ALIST_BASE_URL)
        and ALIST_TOKEN
    ):
        logger.debug(f"为Alist链接添加认证头: {video_source[:70]}...")
        command.extend(["-headers", f"Authorization: {ALIST_TOKEN}"])

    command.extend(["-ss", THUMBNAIL_TIME, "-i", video_source])

    # 仅在提供了 thumbnail_size 时才添加缩放过滤器
    if thumbnail_size:
        logger.debug(f"应用缩放尺寸: {thumbnail_size}")
        command.extend(["-vf", f"scale={thumbnail_size}"])
    else:
        logger.debug("未提供缩放尺寸，将使用源视频分辨率。")

    command.extend(["-vframes", "1", "-q:v", "3", "-y", thumb_path])

    logger.info(f"正在生成缩略图: {thumb_path} (来源: {str(video_source)[:70]}...)")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=90, check=False
        )
        if result.returncode != 0:
            safe_command = (
                " ".join(command).replace(ALIST_TOKEN, "***TOKEN***")
                if ALIST_TOKEN
                else " ".join(command)
            )
            logger.error(
                f"FFmpeg生成缩略图失败: {thumb_path}\n命令: {safe_command}\n错误: {result.stderr}"
            )
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
            return False
        logger.info(f"成功生成缩略图: {thumb_path}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg生成缩略图超时: {thumb_path}")
        return False
    except Exception as e:
        logger.error(f"执行FFmpeg时发生异常 ({thumb_path}): {e}")
        return False


# </editor-fold>


# <editor-fold desc="--- 实时本地监控 (Watchdog) ---">
def process_single_file(absolute_video_path, task_config):
    """处理单个本地视频文件：创建.strm和缩略图。"""
    task_name = task_config["name"]
    monitor_path = task_config["local_monitor_path"]
    relative_path = os.path.relpath(absolute_video_path, monitor_path)
    strm_path, thumb_path = get_strm_and_thumb_paths(relative_path, task_config)

    logger.info(f"任务 '{task_name}': 检测到文件，开始处理: {absolute_video_path}")
    os.makedirs(os.path.dirname(strm_path), exist_ok=True)
    try:
        with open(strm_path, "w", encoding="utf-8") as f:
            f.write(absolute_video_path)
        logger.info(f"任务 '{task_name}': 已创建STRM文件: {strm_path}")
        if thumb_path:
            generate_thumbnail(absolute_video_path, thumb_path, task_config)
    except IOError as e:
        logger.error(f"任务 '{task_name}': 无法写入STRM文件 {strm_path}: {e}")


def delete_generated_files(absolute_video_path, task_config):
    """删除与已删除的本地视频文件对应的.strm和缩略图。"""
    task_name = task_config["name"]
    monitor_path = task_config.get("local_monitor_path")  # Might be called from cleanup
    relative_path = ""

    # Heuristic to figure out relative path from either video path or strm path
    if monitor_path and absolute_video_path.startswith(monitor_path):
        relative_path = os.path.relpath(absolute_video_path, monitor_path)
    elif absolute_video_path.endswith(".strm"):
        base_name, _ = os.path.splitext(absolute_video_path)
        relative_path = os.path.relpath(base_name, task_config["local_strm_dir"])

    strm_path, thumb_path = get_strm_and_thumb_paths(relative_path, task_config)

    # Ensure we use the provided path for strm deletion during cleanup
    strm_to_delete = (
        absolute_video_path if absolute_video_path.endswith(".strm") else strm_path
    )

    logger.info(
        f"任务 '{task_name}': 检测到删除，清理生成文件 (源: {absolute_video_path})"
    )
    for f_path in [strm_to_delete, thumb_path]:
        if f_path and os.path.exists(f_path):
            try:
                os.remove(f_path)
                logger.info(f"任务 '{task_name}': 已删除文件: {f_path}")
            except OSError as e:
                logger.error(f"任务 '{task_name}': 删除文件 {f_path} 失败: {e}")


class LocalVideoEventHandler(FileSystemEventHandler):
    def __init__(self, task_config):
        self.task_config = task_config
        self.video_extensions = [
            ext.strip().lower() for ext in task_config["video_extensions"].split(",")
        ]
        super().__init__()

    def _is_video(self, path):
        return not os.path.isdir(path) and any(
            path.lower().endswith(ext) for ext in self.video_extensions
        )

    def on_created(self, event):
        if self._is_video(event.src_path):
            process_single_file(event.src_path, self.task_config)

    def on_deleted(self, event):
        if self._is_video(event.src_path):
            delete_generated_files(event.src_path, self.task_config)

    def on_moved(self, event):
        if self._is_video(event.src_path):
            delete_generated_files(event.src_path, self.task_config)
        if self._is_video(event.dest_path):
            process_single_file(event.dest_path, self.task_config)


# </editor-fold>


# <editor-fold desc="--- 轮询任务 (Alist & 本地全量扫描) ---">
def get_alist_files(task_config):
    """从Alist获取文件列表。"""
    # (此函数实现与您原始脚本中的 get_alist_files_for_task 相同, 这里为保持完整性而包含)
    base_alist_path = task_config["alist_path"]
    video_extensions = [
        ext.strip().lower() for ext in task_config["video_extensions"].split(",")
    ]
    api_url = urljoin(ALIST_BASE_URL, "/api/fs/list")
    headers = {"Content-Type": "application/json", "Authorization": ALIST_TOKEN or ""}
    files_to_process = {}
    dir_stack = [(base_alist_path, "")]

    while dir_stack:
        current_path, rel_prefix = dir_stack.pop()
        logger.debug(f"任务 '{task_config['name']}': 扫描Alist目录: {current_path}")
        payload = {
            "path": current_path,
            "password": task_config.get("alist_path_password", ""),
            "page": 1,
            "per_page": 0,
        }
        try:
            res = requests.post(api_url, headers=headers, json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            if data.get("code") == 200 and data.get("data"):
                # Fix: Alist API might return 'content': null for empty directories
                content_list = data["data"].get("content") or []
                for item in content_list:
                    name = item.get("name")
                    rel_path = os.path.join(rel_prefix, name).replace("\\", "/")
                    if item.get("is_dir"):
                        dir_stack.append(
                            (
                                os.path.join(current_path, name).replace("\\", "/"),
                                rel_path,
                            )
                        )
                    elif any(name.lower().endswith(ext) for ext in video_extensions):
                        url = item.get("raw_url") or urljoin(
                            ALIST_BASE_URL,
                            "d"
                            + quote(
                                os.path.join(current_path, name).replace("\\", "/"),
                                safe="/",
                            ),
                        )
                        files_to_process[rel_path] = url
        except requests.RequestException as e:
            logger.error(
                f"任务 '{task_config['name']}': 请求Alist API失败 (路径: {current_path}): {e}"
            )
            break
    return files_to_process


def get_local_files(task_config):
    """从本地目录获取文件列表。"""
    monitor_path = task_config["local_monitor_path"]
    video_extensions = [
        ext.strip().lower() for ext in task_config["video_extensions"].split(",")
    ]
    files_to_process = {}
    if not os.path.isdir(monitor_path):
        logger.error(
            f"任务 '{task_config['name']}': 本地路径 '{monitor_path}' 不存在。"
        )
        return files_to_process
    for root, _, files in os.walk(monitor_path):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, monitor_path).replace("\\", "/")
                files_to_process[rel_path] = abs_path
    return files_to_process


def run_full_scan_and_cleanup(task_config, source_files_map):
    """通用全量扫描函数：对比源和目标，进行创建、更新和清理。"""
    task_name = task_config["name"]
    local_strm_base_dir = task_config["local_strm_dir"]

    logger.info(
        f"任务 '{task_name}': 开始全量扫描，发现 {len(source_files_map)} 个源文件。"
    )

    current_strm_files_on_disk = set()
    # 1. 创建或更新文件
    for rel_path, source_url_or_path in source_files_map.items():
        strm_path, thumb_path = get_strm_and_thumb_paths(rel_path, task_config)
        current_strm_files_on_disk.add(strm_path)

        needs_update = True
        if os.path.exists(strm_path):
            try:
                with open(strm_path, "r", encoding="utf-8") as f:
                    if f.read().strip() == source_url_or_path.strip():
                        needs_update = False
            except IOError:
                pass

        if needs_update:
            os.makedirs(os.path.dirname(strm_path), exist_ok=True)
            with open(strm_path, "w", encoding="utf-8") as f:
                f.write(source_url_or_path)
            logger.info(f"任务 '{task_name}': 已创建/更新STRM: {strm_path}")

        if thumb_path:
            generate_thumbnail(source_url_or_path, thumb_path, task_config)

    # 2. 清理失效文件
    logger.info(f"任务 '{task_name}': 开始清理失效文件...")
    if not os.path.isdir(local_strm_base_dir):
        logger.warning(
            f"任务 '{task_name}': STRM目录 '{local_strm_base_dir}' 不存在，跳过清理。"
        )
        return

    for root, _, files in os.walk(local_strm_base_dir):
        for file in files:
            if file.lower().endswith(".strm"):
                path_on_disk = os.path.join(root, file)
                if path_on_disk not in current_strm_files_on_disk:
                    # 使用 strm 路径调用删除函数，它会计算出对应的缩略图路径
                    delete_generated_files(path_on_disk, task_config)

    logger.info(f"任务 '{task_name}': 全量扫描和清理完成。")


def run_task(task_config, file_fetcher_func):
    """轮询任务的通用执行器。"""
    logger.info(f"--- 开始轮询任务: '{task_config['name']}' ---")
    source_files = file_fetcher_func(task_config)
    run_full_scan_and_cleanup(task_config, source_files)
    logger.info(f"--- 任务 '{task_config['name']}' 完成 ---")


# </editor-fold>


# <editor-fold desc="--- 配置加载与主程序 ---">
def load_config(config_file="config.ini"):
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_file):
        print(f"致命错误: 配置文件 {config_file} 未找到。")
        exit(1)
    config.read(config_file, encoding="utf-8")

    global ALIST_BASE_URL, ALIST_TOKEN, FFMPEG_PATH, THUMBNAIL_TIME
    try:
        ALIST_BASE_URL = config.get("alist_global", "base_url").rstrip("/")
        ALIST_TOKEN = config.get("alist_global", "token", fallback="").strip() or None
        FFMPEG_PATH = config.get("ffmpeg_global", "path", fallback="ffmpeg")
        THUMBNAIL_TIME = config.get(
            "ffmpeg_global", "thumbnail_time", fallback="00:00:10"
        )
        log_level = config.get("logging_global", "level", fallback="INFO").upper()
        log_format = config.get(
            "logging_global",
            "format",
            fallback="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO), format=log_format
        )
    except Exception as e:
        logging.error(f"加载全局配置失败: {e}", exc_info=True)
        exit(1)

    alist_tasks, scheduled_local_tasks, live_local_tasks = [], [], []
    for section in config.sections():
        if not section.startswith(("sync_task_", "local_monitor_task_")):
            continue
        try:
            if not config.getboolean(section, "enabled", fallback=True):
                logger.debug(f"任务 '{section}' 已禁用，跳过。")
                continue

            common_config = {
                "name": section.split("_", 2)[-1],
                "local_strm_dir": config.get(section, "local_strm_dir"),
                "local_thumb_dir": config.get(section, "local_thumb_dir", fallback=None)
                or None,
                "video_extensions": config.get(section, "video_extensions"),
                "schedule_mode": config.get(section, "schedule_mode").lower(),
                "run_on_startup": config.getboolean(
                    section, "run_on_startup", fallback=False
                ),
                # 新增的任务级缩略图配置
                "generate_thumbnails": config.getboolean(
                    section, "generate_thumbnails", fallback=False
                ),
                "thumbnail_size": config.get(
                    section, "thumbnail_size", fallback="320x180"
                ),
                "thumbnail_format": config.get(
                    section, "thumbnail_format", fallback="jpg"
                ),
                "thumbnail_suffix": config.get(
                    section, "thumbnail_suffix", fallback="poster"
                ).strip(),
            }

            # --- 新增: 自动创建任务目录 ---
            if common_config.get("local_strm_dir"):
                os.makedirs(common_config["local_strm_dir"], exist_ok=True)
                logger.debug(f"确保STRM目录存在: {common_config['local_strm_dir']}")
            if common_config.get("local_thumb_dir"):
                os.makedirs(common_config["local_thumb_dir"], exist_ok=True)
                logger.debug(f"确保Thumb目录存在: {common_config['local_thumb_dir']}")
            # --- 结束 ---

            if common_config["schedule_mode"] != "live":
                common_config["schedule_interval_minutes"] = config.getint(
                    section, "schedule_interval_minutes", fallback=0
                )
                common_config["schedule_daily_at_time"] = config.get(
                    section, "schedule_daily_at_time", fallback=""
                ).strip()

            if section.startswith("sync_task_"):
                common_config["alist_path"] = config.get(section, "alist_path")
                common_config["alist_path_password"] = config.get(
                    section, "alist_path_password", fallback=""
                )
                alist_tasks.append(common_config)
            elif section.startswith("local_monitor_task_"):
                common_config["local_monitor_path"] = config.get(
                    section, "local_monitor_path"
                )
                if common_config["schedule_mode"] == "live":
                    live_local_tasks.append(common_config)
                else:
                    scheduled_local_tasks.append(common_config)
            logger.info(
                f"成功加载任务: '{common_config['name']}' (模式: {common_config['schedule_mode']})"
            )
        except Exception as e:
            logger.error(f"加载任务 '{section}' 失败: {e}", exc_info=True)

    return alist_tasks, scheduled_local_tasks, live_local_tasks


def scheduler_loop():
    """在独立线程中运行所有定时任务。"""
    while not stop_scheduler_thread.is_set():
        schedule.run_pending()
        time.sleep(1)
    logger.info("调度器线程已停止。")


if __name__ == "__main__":
    alist_tasks, scheduled_local, live_local = load_config()

    # 1. 设置轮询任务
    for task in alist_tasks:
        if task["run_on_startup"]:
            threading.Thread(
                target=run_task,
                args=(task, get_alist_files),
                name=f"{task['name']}-Startup",
            ).start()
        job = partial(run_task, task, get_alist_files)
        if (
            task["schedule_mode"] == "interval"
            and task["schedule_interval_minutes"] > 0
        ):
            schedule.every(task["schedule_interval_minutes"]).minutes.do(job)
        elif task["schedule_mode"] == "daily" and task["schedule_daily_at_time"]:
            schedule.every().day.at(task["schedule_daily_at_time"]).do(job)

    for task in scheduled_local:
        if task["run_on_startup"]:
            threading.Thread(
                target=run_task,
                args=(task, get_local_files),
                name=f"{task['name']}-Startup",
            ).start()
        job = partial(run_task, task, get_local_files)
        if (
            task["schedule_mode"] == "interval"
            and task["schedule_interval_minutes"] > 0
        ):
            schedule.every(task["schedule_interval_minutes"]).minutes.do(job)
        elif task["schedule_mode"] == "daily" and task["schedule_daily_at_time"]:
            schedule.every().day.at(task["schedule_daily_at_time"]).do(job)

    # 2. 启动轮询任务线程 (如果存在)
    if schedule.get_jobs():
        scheduler_thread = threading.Thread(
            target=scheduler_loop, name="SchedulerThread"
        )
        scheduler_thread.start()
        logger.info("轮询任务调度器已在后台线程启动。")
    else:
        scheduler_thread = None

    # 3. 设置并启动实时监控任务
    observers = []
    for task in live_local:
        monitor_path = task["local_monitor_path"]
        if not os.path.isdir(monitor_path):
            logger.error(
                f"实时监控任务 '{task['name']}' 的路径 '{monitor_path}' 无效，跳过此任务。"
            )
            continue

        if task["run_on_startup"]:
            # 对于live模式，启动时执行一次全量扫描是个好主意
            threading.Thread(
                target=run_task,
                args=(task, get_local_files),
                name=f"{task['name']}-Startup",
            ).start()

        event_handler = LocalVideoEventHandler(task)
        observer = Observer()
        observer.schedule(event_handler, monitor_path, recursive=True)
        observer.start()
        observers.append(observer)
        logger.info(
            f"实时监控任务 '{task['name']}' 已启动，正在监控路径: {monitor_path}"
        )

    # 4. 主循环与优雅退出
    logger.info("脚本正在运行... 按 Ctrl+C 退出。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭所有服务...")
        if scheduler_thread and scheduler_thread.is_alive():
            stop_scheduler_thread.set()
            scheduler_thread.join()
        for observer in observers:
            observer.stop()
            observer.join()
        logger.info("所有服务已安全关闭。再见！")
