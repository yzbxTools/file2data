"""
现代化文件浏览器

需求:
1. 通过argparse 或 网页ui, 指定 BASE_DIR
2. 支持浏览文件夹和文件, 并显示文件类型
3. 支持目录跳转，如回退到上一级目录, 或者跳转到根目录
4. 支持文件搜索功能
5. 支持分页浏览, 同时预览页面下所有图片或视频
6. 支持多种文件类型预览
7. 现代化UI界面

使用方法:

streamlit run view_file.py -- --base_dir /path/to/your/directory

"""

import streamlit as st
import os
import argparse
from pathlib import Path
from rapidfuzz import fuzz
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="现代化文件浏览器")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="",
        help="基础目录路径"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="服务端口"
    )
    return parser.parse_args()

def get_file_icon(filename):
    """根据文件扩展名返回对应的图标"""
    ext = Path(filename).suffix.lower()
    
    # 图片文件
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']:
        return "🖼️"
    # 视频文件
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']:
        return "🎥"
    # 音频文件
    elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma']:
        return "🎵"
    # 文档文件
    elif ext in ['.pdf', '.doc', '.docx', '.txt', '.rtf']:
        return "📄"
    elif ext in ['.xls', '.xlsx', '.csv']:
        return "📊"
    elif ext in ['.ppt', '.pptx']:
        return "📈"
    # 压缩文件
    elif ext in ['.zip', '.rar', '.7z', '.tar', '.gz']:
        return "📦"
    # 代码文件
    elif ext in ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h']:
        return "💻"
    # 其他文件
    else:
        return "📄"

def get_file_size(file_path):
    """获取文件大小的人类可读格式"""
    try:
        size = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except:
        return "未知"

def search_files(directory, search_term):
    """在目录中搜索文件"""
    if not search_term:
        return []
    
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fuzz.partial_ratio(search_term.lower(), file.lower()) > 60:
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                results.append(rel_path)
    return results[:50]  # 限制结果数量

def display_image(file_path, width=200):
    """显示图片文件"""
    try:
        image = Image.open(file_path)
        st.image(image, width=width, caption=os.path.basename(file_path))
    except Exception as e:
        st.error(f"无法显示图片: {e}")

def display_video(file_path):
    """显示视频文件"""
    try:
        with open(file_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
    except Exception as e:
        st.error(f"无法显示视频: {e}")

def display_text(file_path):
    """显示文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        st.text_area("文件内容", content, height=300)
    except Exception as e:
        st.error(f"无法读取文本文件: {e}")

def get_media_files(entries, full_path):
    """获取当前页面的媒体文件（图片和视频）"""
    media_files = []
    for name, is_dir, size in entries:
        if not is_dir:
            ext = Path(name).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']:
                media_files.append((name, 'image', os.path.join(full_path, name)))
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']:
                media_files.append((name, 'video', os.path.join(full_path, name)))
    return media_files

def display_media_preview(media_files, preview_mode="grid"):
    """批量预览媒体文件"""
    if not media_files:
        st.info("当前页面没有可预览的媒体文件")
        return
    
    st.subheader(f"🖼️ 媒体预览 ({len(media_files)} 个文件)")
    
    if preview_mode == "grid":
        # 网格布局预览
        cols = st.columns(3)  # 3列布局
        for i, (name, file_type, file_path) in enumerate(media_files):
            col_idx = i % 3
            with cols[col_idx]:
                st.write(f"**{name}**")
                if file_type == 'image':
                    try:
                        image = Image.open(file_path)
                        st.image(image, width=200, caption=name)
                    except Exception as e:
                        st.error(f"无法显示图片: {e}")
                elif file_type == 'video':
                    try:
                        with open(file_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                    except Exception as e:
                        st.error(f"无法显示视频: {e}")
    
    elif preview_mode == "list":
        # 列表布局预览
        for name, file_type, file_path in media_files:
            with st.expander(f"{'🖼️' if file_type == 'image' else '🎥'} {name}"):
                if file_type == 'image':
                    try:
                        image = Image.open(file_path)
                        st.image(image, width=400, caption=name)
                    except Exception as e:
                        st.error(f"无法显示图片: {e}")
                elif file_type == 'video':
                    try:
                        with open(file_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                    except Exception as e:
                        st.error(f"无法显示视频: {e}")

def main():
    # 设置页面配置
    st.set_page_config(
        page_title="现代化文件浏览器",
        page_icon="📁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📁 现代化文件浏览器")
    st.markdown("---")
    
    # 获取命令行参数
    args = get_args()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("📂 目录设置")
        
        # 基础目录输入
        base_dir = st.text_input(
            "基础目录路径:",
            value=args.base_dir,
            help="请输入要浏览的目录路径"
        )
        
        if st.button("选择目录", help="选择要浏览的目录"):
            # 这里可以添加目录选择器
            pass
        
        st.markdown("---")
        
        # 搜索功能
        st.header("🔍 搜索文件")
        search_term = st.text_input(
            "搜索文件:",
            placeholder="输入文件名进行模糊搜索",
            help="支持模糊搜索文件名"
        )
        
        st.markdown("---")
        
        # 过滤选项
        st.header("🔧 过滤选项")
        show_folders = st.checkbox("显示文件夹", value=True)
        show_files = st.checkbox("显示文件", value=True)
        
        # 文件类型过滤
        file_types = st.multiselect(
            "文件类型过滤:",
            ["图片", "视频", "音频", "文档", "压缩包", "代码"],
            default=["图片", "视频", "音频", "文档", "压缩包", "代码"]
        )
        
        # 分页设置
        st.markdown("---")
        st.header("📄 分页设置")
        items_per_page = st.slider("每页显示项目数:", 10, 100, 20)
        
        # 媒体预览设置 - 新增功能
        st.markdown("---")
        st.header("🖼️ 媒体预览设置")
        enable_media_preview = st.checkbox("启用媒体预览", value=False, help="同时预览当前页面的所有图片和视频")
        if enable_media_preview:
            preview_mode = st.selectbox(
                "预览模式:",
                ["grid", "list"],
                format_func=lambda x: "网格布局" if x == "grid" else "列表布局",
                help="选择媒体文件的预览布局方式"
            )
            preview_position = st.selectbox(
                "预览位置:",
                ["top", "bottom"],
                format_func=lambda x: "页面顶部" if x == "top" else "页面底部",
                help="选择媒体预览在页面中的显示位置"
            )
    
    # 主内容区域
    if base_dir and os.path.exists(base_dir):
        # 获取当前路径
        current_path = st.session_state.get('current_path', '')
        full_path = os.path.join(base_dir, current_path)
        
        # 导航栏 - 新增功能
        st.subheader("🧭 导航")
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
        
        with nav_col1:
            if st.button(" 返回根目录", help="跳转到根目录"):
                st.session_state['current_path'] = ''
                st.rerun()
        
        with nav_col2:
            if current_path and st.button("⬆️ 返回上级目录", help="回退到上一级目录"):
                # 获取上级目录路径
                parent_path = os.path.dirname(current_path)
                st.session_state['current_path'] = parent_path
                st.rerun()
        
        with nav_col3:
            st.write(f"当前路径: `{current_path or '根目录'}`")
        
        st.markdown("---")
        
        # 面包屑导航
        if current_path:
            breadcrumbs = current_path.split('/')
            breadcrumb_html = "📁 "
            for i, crumb in enumerate(breadcrumbs):
                if crumb:
                    path_to_crumb = '/'.join(breadcrumbs[:i+1])
                    breadcrumb_html += f"<a href='?path={path_to_crumb}'>{crumb}</a> / "
            st.markdown(breadcrumb_html, unsafe_allow_html=True)
        
        # 搜索结果显示
        if search_term:
            st.subheader(f"🔍 搜索结果: '{search_term}'")
            search_results = search_files(full_path, search_term)
            if search_results:
                for result in search_results:
                    file_path = os.path.join(full_path, result)
                    icon = get_file_icon(result)
                    size = get_file_size(file_path)
                    st.write(f"{icon} {result} ({size})")
            else:
                st.info("未找到匹配的文件")
        else:
            # 正常目录浏览
            try:
                entries = []
                for item in os.listdir(full_path):
                    item_path = os.path.join(full_path, item)
                    is_dir = os.path.isdir(item_path)
                    
                    if is_dir and show_folders:
                        entries.append((item, True, None))
                    elif not is_dir and show_files:
                        entries.append((item, False, get_file_size(item_path)))
                
                # 排序：文件夹在前，然后按名称排序
                entries.sort(key=lambda x: (not x[1], x[0].lower()))
                
                # 分页
                total_items = len(entries)
                total_pages = (total_items + items_per_page - 1) // items_per_page
                
                if total_pages > 1:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        page = st.selectbox("页码", range(1, total_pages + 1), index=0)
                else:
                    page = 1
                
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_entries = entries[start_idx:end_idx]
                
                # 媒体预览功能 - 新增
                if enable_media_preview and preview_position == "top":
                    media_files = get_media_files(page_entries, full_path)
                    if media_files:
                        display_media_preview(media_files, preview_mode)
                        st.markdown("---")
                
                # 显示条目
                for name, is_dir, size in page_entries:
                    if is_dir:
                        icon = "📁"
                        if st.button(f"{icon} {name}/", key=f"dir_{name}"):
                            st.session_state['current_path'] = os.path.join(current_path, name).lstrip('/')
                            st.rerun()
                    else:
                        icon = get_file_icon(name)
                        file_path = os.path.join(full_path, name)
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            if st.button(f"{icon} {name}", key=f"file_{name}"):
                                # 文件预览
                                ext = Path(name).suffix.lower()
                                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                                    display_image(file_path)
                                elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                                    display_video(file_path)
                                elif ext in ['.txt', '.py', '.js', '.html', '.css', '.json']:
                                    display_text(file_path)
                                else:
                                    st.info(f"文件类型 {ext} 暂不支持预览")
                        
                        with col2:
                            st.write(size if size else "")
                        
                        with col3:
                            # 下载链接
                            with open(file_path, "rb") as f:
                                file_bytes = f.read()
                            st.download_button(
                                label="下载",
                                data=file_bytes,
                                file_name=name,
                                key=f"download_{name}"
                            )
                
                # 媒体预览功能 - 底部显示
                if enable_media_preview and preview_position == "bottom":
                    media_files = get_media_files(page_entries, full_path)
                    if media_files:
                        st.markdown("---")
                        display_media_preview(media_files, preview_mode)
                
                # 显示统计信息
                st.markdown("---")
                st.write(f"总计: {total_items} 个项目 (第 {page}/{total_pages} 页)")
                
            except PermissionError:
                st.error("没有权限访问此目录")
            except Exception as e:
                st.error(f"访问目录时出错: {e}")
    else:
        st.info("请在侧边栏输入有效的目录路径")
    
    # 页脚
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("""
    1. **设置目录**: 在侧边栏输入要浏览的目录路径
    2. **搜索文件**: 使用模糊搜索功能快速找到文件
    3. **浏览文件**: 点击文件夹进入子目录，点击文件预览内容
    4. **过滤选项**: 使用侧边栏的过滤选项控制显示内容
    5. **分页浏览**: 当文件较多时，使用分页功能浏览
    6. **媒体预览**: 启用媒体预览功能可同时查看当前页面的所有图片和视频
    7. **文件下载**: 点击下载按钮下载文件
    """)

if __name__ == '__main__':
    main()