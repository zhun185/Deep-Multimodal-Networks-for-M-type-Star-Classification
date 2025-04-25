import os
import time
import requests
import pandas as pd
import re # Import regular expressions module

# --- 配置 ---
CSV_FILE = 'objects.csv'  # 输入的 CSV 文件名，必须包含 subclass 列
IMAGE_DIR = 'images'      # 保存图像的根文件夹
SPECTRA_DIR = 'spectra'    # 保存光谱的根文件夹
DATA_RELEASE = 'dr16'     # SDSS 数据版本
IMG_WIDTH = 128           # 图像宽度 (像素)
IMG_HEIGHT = 128          # 图像高度 (像素)
IMG_SCALE = 0.2           # 图像缩放比例 (arcsec/pixel, 可调整以匹配128x128视场)
DOWNLOAD_IMAGES = True    # 是否下载图像
DOWNLOAD_SPECTRA = True   # 是否下载光谱
# --- 配置结束 ---

# 创建根输出文件夹 (如果不存在)
if DOWNLOAD_IMAGES and not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
if DOWNLOAD_SPECTRA and not os.path.exists(SPECTRA_DIR):
    os.makedirs(SPECTRA_DIR)

# --- 函数定义 ---
def download_file(url, save_path):
    """下载文件并保存到指定路径"""
    try:
        # 确保保存文件的目录存在
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
                print(f"  创建目录: {save_dir}")
            except OSError as e:
                print(f"  创建目录失败: {save_dir} - {e}")
                return False # 无法创建目录，下载失败

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # 如果请求失败则引发 HTTPError
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # print(f"  下载成功: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  下载失败: {url} - {e}")
        return False

# --- 主程序 ---
print(f"正在从 {CSV_FILE} 读取目标列表...")
try:
    # 指定列的数据类型，确保 plate, mjd, fiberID, name 被读取为整数
    # 将 subclass 读取为字符串
    dtypes = {
        'name': 'int64',
        'plate': 'int',
        'mjd': 'int',
        'fiberID': 'int',
        'subclass': 'str' # 读取 subclass 为字符串
    }
    df = pd.read_csv(CSV_FILE, dtype=dtypes)
    # 填充可能存在的 NaN/空 subclass 值，以防出错
    df['subclass'] = df['subclass'].fillna('Unknown').astype(str)
    print(f"找到 {len(df)} 个目标.")
except FileNotFoundError:
    print(f"错误: 未找到文件 {CSV_FILE}。请确保文件存在于脚本同目录下。")
    exit()
except KeyError:
    print(f"错误: CSV 文件 {CSV_FILE} 缺少 'subclass' 列。请确保你下载了包含该列的 CSV 文件。")
    exit()
except Exception as e:
    print(f"读取 CSV 文件时出错: {e}")
    exit()

# 检查必需的列是否存在 (现在包括 subclass)
required_cols = ['name', 'ra', 'dec', 'subclass']
if DOWNLOAD_SPECTRA:
    required_cols.extend(['plate', 'mjd', 'fiberID'])

if not all(col in df.columns for col in required_cols):
    print(f"错误: CSV 文件缺少必需的列。需要: {required_cols}")
    print(f"当前列: {df.columns.tolist()}")
    exit()


# 遍历每个目标进行下载
for index, row in df.iterrows():
    name = row['name']
    ra = row['ra']
    dec = row['dec']
    subclass = row['subclass'].strip() # 获取 subclass 并去除首尾空格

    print(f"\n正在处理目标 {index + 1}/{len(df)}: name={name}, ra={ra}, dec={dec}, subclass={subclass}")

    # 验证 subclass 是否为有效的 M 型 (或其他可接受的格式)
    # 使用正则表达式匹配 M 后跟数字，或标记为 Unknown
    if not re.match(r"^M[0-9]$", subclass) and subclass != 'Unknown':
        print(f"  警告：无效或非预期的 subclass '{subclass}'，将放入 'Unknown' 文件夹。")
        subclass_folder = 'Unknown'
    elif subclass == 'Unknown':
         print(f"  警告：subclass 为空或未知，将放入 'Unknown' 文件夹。")
         subclass_folder = 'Unknown'
    else:
        subclass_folder = subclass # 使用有效的 subclass 作为文件夹名

    # --- 下载光谱 (优先处理，因为如果光谱失败则跳过图像) ---
    downloaded_spectrum = False
    spec_filename = "" # 初始化

    if DOWNLOAD_SPECTRA:
        try:
            plate = int(row['plate'])
            mjd = int(row['mjd'])
            fiberID = int(row['fiberID'])
        except ValueError as e:
            print(f"  错误：无法将 plate/mjd/fiberID 转换为整数，跳过此目标。")
            continue

        run2d_eboss = 'v5_13_0'
        run2d_boss = 'v5_13_2'
        run2d_sdss = '26'

        spec_base_filename = f"spec-{plate:04d}-{mjd:5d}-{fiberID:04d}.fits"
        # 构建包含 subclass 的完整光谱文件路径
        spec_filename = os.path.join(SPECTRA_DIR, subclass_folder, spec_base_filename)

        if os.path.exists(spec_filename):
             print(f"  光谱已存在，跳过: {spec_filename}")
             downloaded_spectrum = True # 标记为已下载（或已存在）
        else:
            # --- 尝试不同的下载路径 --- #
            # 1. 尝试 eBOSS 路径
            spec_url_eboss = f"https://data.sdss.org/sas/{DATA_RELEASE}/eboss/spectro/redux/{run2d_eboss}/spectra/{plate:04d}/{spec_base_filename}"
            print(f"  尝试下载光谱 (eBOSS 路径): {spec_url_eboss}")
            if download_file(spec_url_eboss, spec_filename):
                 downloaded_spectrum = True

            # 2. 如果失败，尝试 BOSS 路径
            if not downloaded_spectrum:
                spec_url_boss = f"https://data.sdss.org/sas/{DATA_RELEASE}/boss/spectro/redux/{run2d_boss}/spectra/{plate:04d}/{spec_base_filename}"
                print(f"  尝试下载光谱 (BOSS 路径): {spec_url_boss}")
                if download_file(spec_url_boss, spec_filename):
                    downloaded_spectrum = True

            # 3. 如果仍然失败，尝试更通用的 SDSS 路径 (使用 run2d=26)
            if not downloaded_spectrum:
                spec_url_sdss = f"https://data.sdss.org/sas/{DATA_RELEASE}/sdss/spectro/redux/{run2d_sdss}/spectra/{plate:04d}/{spec_base_filename}"
                print(f"  尝试下载光谱 (SDSS 路径, run2d={run2d_sdss}): {spec_url_sdss}")
                if download_file(spec_url_sdss, spec_filename):
                     downloaded_spectrum = True

            # 4. 如果仍然失败，尝试更通用的 SDSS 路径 (使用 run2d=v5_13_0)
            if not downloaded_spectrum:
                 spec_url_sdss_alt = f"https://data.sdss.org/sas/{DATA_RELEASE}/sdss/spectro/redux/{run2d_eboss}/spectra/{plate:04d}/{spec_base_filename}"
                 print(f"  尝试下载光谱 (SDSS 路径, run2d={run2d_eboss}): {spec_url_sdss_alt}")
                 if download_file(spec_url_sdss_alt, spec_filename):
                      downloaded_spectrum = True

            # 所有尝试后进行延时
            if downloaded_spectrum:
                time.sleep(0.5)
            else:
                print(f"  警告：所有尝试的路径均无法下载光谱文件 {spec_base_filename}")
                time.sleep(0.2)

    # --- 下载图像 (仅当光谱下载成功或光谱下载被禁用时进行) ---
    # 如果光谱下载被禁用，或者光谱下载成功了，才继续下载图像
    should_download_image = not DOWNLOAD_SPECTRA or downloaded_spectrum

    if DOWNLOAD_IMAGES and should_download_image:
        # 构建包含 subclass 的完整图像文件路径
        img_filename = os.path.join(IMAGE_DIR, subclass_folder, f"{name}_image.jpg")
        if os.path.exists(img_filename):
            print(f"  图像已存在，跳过: {img_filename}")
        else:
            img_url = f"https://skyserver.sdss.org/{DATA_RELEASE}/SkyServerWS/ImgCutout/getjpeg"
            img_params = f"?ra={ra}&dec={dec}&scale={IMG_SCALE}&width={IMG_WIDTH}&height={IMG_HEIGHT}&opt="
            full_img_url = img_url + img_params
            print(f"  尝试下载图像: {full_img_url}")
            if download_file(full_img_url, img_filename):
                time.sleep(0.5) # 成功下载后延时
            else:
                 # 如果图像下载失败，可以选择是否删除已下载的光谱（如果需要严格配对）
                 # print(f"  警告：图像下载失败，但光谱可能已下载。")
                 pass

    elif DOWNLOAD_IMAGES and not should_download_image:
         print(f"  光谱下载失败或跳过，因此跳过下载图像。")


print("\n下载完成!")

if not DOWNLOAD_SPECTRA:
    print("\n请注意：光谱文件未下载，因为脚本配置 DOWNLOAD_SPECTRA=False。")
