import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import tkinter as tk
from tkinter import filedialog

def plot_spectrum(filepath):
    """Reads FITS spectrum file and plots Flux vs Wavelength.
    读取 FITS 光谱文件并绘制 Flux vs Wavelength 图。
    """
    try:
        with fits.open(filepath) as hdul:
            print(f"Successfully opened FITS file: {filepath}")
            # print(f"成功打开 FITS 文件: {filepath}")
            # 打印 HDU 列表信息 (可选)
            # hdul.info()

            # 假设光谱数据在 HDU 1 (对于 SDSS spec-*.fits 文件通常如此)
            # HDU 0 通常是空的 PrimaryHDU
            if len(hdul) < 2:
                print(f"Error: File {filepath} does not contain enough Header-Data Units (HDUs). Expected at least 2.")
                # print(f"错误: 文件 {filepath} 不包含足够的数据单元 (HDU)。期望至少有 2 个 HDU。")
                return

            header = hdul[1].header
            data = hdul[1].data

            # --- 提取数据 ---
            try:
                flux = data['flux'].astype(np.float32)
            except KeyError:
                print(f"Error: 'flux' column not found in HDU 1. Please check the FITS file structure.")
                # print(f"错误: 在 HDU 1 中未找到 'flux' 列。请检查 FITS 文件结构。")
                return

            # --- 计算波长 --- #
            # SDSS 光谱通常使用对数波长标尺
            # 波长 = 10^(CRVAL1 + CDELT1 * (像素索引 + 1))
            # 注意：像素索引在 FITS 标准中通常是 1-based，但在 NumPy 中是 0-based
            # 检查常见的关键字：CRVAL1 (起始值), CDELT1 或 CD1_1 (步长)
            try:
                log_start_wl = header['CRVAL1'] # 对数坐标下的起始波长
                # 优先使用 CD1_1，如果不存在则尝试 CDELT1
                if 'CD1_1' in header:
                    log_dwl = header['CD1_1'] # 对数坐标下的波长步长
                elif 'CDELT1' in header:
                    log_dwl = header['CDELT1']
                else:
                     raise KeyError("Wavelength step keyword (CD1_1 or CDELT1) not found") # 未找到波长步长关键字

                # 创建像素索引数组 (0-based for numpy)
                pixels = np.arange(len(flux))
                # 计算每个像素对应的波长 (Ångströms)
                wavelength = 10**(log_start_wl + pixels * log_dwl)

            except KeyError as e:
                print(f"Error: Missing keyword(s) required for wavelength calculation in FITS header: {e}. Cannot plot wavelength.")
                # print(f"错误: FITS 头文件中缺少计算波长所需的关键字: {e}。无法绘制波长图。")
                # 可以选择只绘制 Flux vs Pixel Index
                wavelength = np.arange(len(flux)) # 使用像素索引作为 x 轴
                x_label = "Pixel Index" # "像素索引"
            except Exception as e:
                 print(f"Error calculating wavelength: {e}")
                 # print(f"计算波长时出错: {e}")
                 return
            else:
                 x_label = "Wavelength (Ångström)" # "波长 (Ångström)"

            # --- 绘图 --- #
            plt.figure(figsize=(12, 6))
            plt.plot(wavelength, flux, label='Flux', color='royalblue', linewidth=1) # label='通量 (Flux)'

            # --- (可选) 绘制其他信息 --- #
            # 例如，绘制模型拟合 (如果存在)
            if 'model' in data.names:
                 try:
                     model_flux = data['model'].astype(np.float32)
                     plt.plot(wavelength, model_flux, label='Model Fit', color='red', linestyle='--', linewidth=1) # label='模型拟合 (Model)'
                 except KeyError:
                     pass # 忽略错误，如果列不存在

            # 例如，根据掩码标记坏点 (可选, 示例)
            # if 'and_mask' in data.names and data['and_mask'].any():
            #     bad_pixels = data['and_mask'] != 0
            #     plt.scatter(wavelength[bad_pixels], flux[bad_pixels], color='orange', marker='x', s=10, label='Bad Pixels (Masked)') # label='坏点 (Masked)'

            # --- 添加图表元素 --- #
            plt.xlabel(x_label)
            plt.ylabel("Flux") # "通量 (Flux)"
            # 从头文件中提取信息添加到标题
            object_name = header.get('OBJECT', 'Unknown Object') # '未知目标'
            plate = header.get('PLATEID', 'N/A')
            mjd = header.get('MJD', 'N/A')
            fiber = header.get('FIBERID', 'N/A')
            # title = f"光谱: {object_name} (Plate={plate}, MJD={mjd}, Fiber={fiber})\n{os.path.basename(filepath)}"
            title = f"Spectrum: {object_name} (Plate={plate}, MJD={mjd}, Fiber={fiber})\n{os.path.basename(filepath)}"
            plt.title(title)
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.show()

    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        # print(f"错误: 文件未找到 {filepath}")
    except Exception as e:
        print(f"Error opening or processing FITS file: {e}")
        # print(f"打开或处理 FITS 文件时发生错误: {e}")

if __name__ == "__main__":
    # 创建一个隐藏的 Tkinter 根窗口
    root = tk.Tk()
    root.withdraw()

    # 打开文件选择对话框
    print("Please select a .fits spectrum file to visualize...")
    # print("请选择一个 .fits 光谱文件进行可视化...")
    file_path = filedialog.askopenfilename(
        title="Select FITS Spectrum File", # "选择 FITS 光谱文件"
        filetypes=(("FITS Files", "*.fits *.fit"), ("All Files", "*.*")) # "FITS 文件", "所有文件"
    )

    # 如果用户选择了文件，则绘制光谱
    if file_path:
        plot_spectrum(file_path)
    else:
        print("No file selected.")
        # print("未选择文件。") 