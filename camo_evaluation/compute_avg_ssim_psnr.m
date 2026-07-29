% 交互式选择五张图像并计算掩膜区域内的平均SSIM与PSNR
% 要求：图像分辨率一致，掩膜为二值图（白色正方形区域），其他图像为彩色
% 输出：图一 vs 图二、图一 vs 图三、图一 vs 图四 在掩膜区域的平均SSIM和PSNR

clear; clc; close all;

%% 1. 交互式选择五张图像文件
fprintf('请选择图一（彩色图像）...\n');
[file1, path1] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', '图像文件'}, '选择图一');
if isequal(file1, 0)
    error('未选择图一，程序终止。');
end
img1 = imread(fullfile(path1, file1));

fprintf('请选择图二（彩色图像）...\n');
[file2, path2] = uigetfile({'*.jpg;*.png;*.bmp;*.tif'}, '选择图二');
if isequal(file2, 0)
    error('未选择图二，程序终止。');
end
img2 = imread(fullfile(path2, file2));

fprintf('请选择图三（彩色图像）...\n');
[file3, path3] = uigetfile({'*.jpg;*.png;*.bmp;*.tif'}, '选择图三');
if isequal(file3, 0)
    error('未选择图三，程序终止。');
end
img3 = imread(fullfile(path3, file3));

fprintf('请选择图四（彩色图像）...\n');
[file4, path4] = uigetfile({'*.jpg;*.png;*.bmp;*.tif'}, '选择图四');
if isequal(file4, 0)
    error('未选择图四，程序终止。');
end
img4 = imread(fullfile(path4, file4));

fprintf('请选择掩膜图像（二值图，白色正方形区域）...\n');
[fileM, pathM] = uigetfile({'*.jpg;*.png;*.bmp;*.tif'}, '选择掩膜');
if isequal(fileM, 0)
    error('未选择掩膜，程序终止。');
end
imgM = imread(fullfile(pathM, fileM));

%% 2. 检查图像分辨率是否一致
sz1 = size(img1);
sz2 = size(img2);
sz3 = size(img3);
sz4 = size(img4);
szM = size(imgM);

if ~isequal(sz1(1:2), sz2(1:2), sz3(1:2), sz4(1:2), szM(1:2))
    error('所有图像的空间分辨率（高度和宽度）必须一致！');
end

if size(img1, 3) ~= 3 || size(img2, 3) ~= 3 || size(img3, 3) ~= 3 || size(img4, 3) ~= 3
    error('图一至图四必须是彩色图像（3通道）。');
end

if size(imgM, 3) > 1
    imgM = rgb2gray(imgM);
    warning('掩膜图像为彩色，已转换为灰度。');
end

%% 3. 预处理：转换为double类型 [0,1] 范围
img1_d = im2double(img1);
img2_d = im2double(img2);
img3_d = im2double(img3);
img4_d = im2double(img4);

% 掩膜二值化（阈值0.5）
threshold = 0.5;
mask = imbinarize(im2double(imgM), threshold);  % 得到logical掩膜

%% 4. 提取掩膜中的各个连通区域（正方形）
[L, numRegions] = bwlabel(mask, 8);
fprintf('检测到 %d 个连通区域（正方形区域）。\n', numRegions);

if numRegions == 0
    error('掩膜中未找到任何白色区域，请检查掩膜图像。');
end

%% 5. 计算三对图像的SSIM全局质量图（基于灰度）
% 将彩色图像转为灰度，用于SSIM计算
img1_gray = rgb2gray(img1_d);
img2_gray = rgb2gray(img2_d);
img3_gray = rgb2gray(img3_d);
img4_gray = rgb2gray(img4_d);

fprintf('正在计算SSIM质量图（图一 vs 图二）...\n');
[~, ssimmap12] = ssim(img1_gray, img2_gray);
fprintf('正在计算SSIM质量图（图一 vs 图三）...\n');
[~, ssimmap13] = ssim(img1_gray, img3_gray);
fprintf('正在计算SSIM质量图（图一 vs 图四）...\n');
[~, ssimmap14] = ssim(img1_gray, img4_gray);

%% 6. 初始化存储每个区域指标的数组
% 每一行对应一个区域，列依次为：SSIM_12, SSIM_13, SSIM_14, PSNR_12, PSNR_13, PSNR_14
regionMetrics = zeros(numRegions, 6);

%% 7. 遍历每个区域，计算指标
for k = 1:numRegions
    currentMask = (L == k);  % 当前区域的逻辑掩膜
    
    % --- SSIM：对质量图取掩膜内平均 ---
    ssim_12 = mean(ssimmap12(currentMask));
    ssim_13 = mean(ssimmap13(currentMask));
    ssim_14 = mean(ssimmap14(currentMask));
    
    % --- PSNR：基于掩膜内所有像素的MSE（彩色三通道平均）---
    idx = find(currentMask);  % 线性索引
    
    % 提取各通道掩膜内像素
    r1 = img1_d(:,:,1); r1_masked = r1(idx);
    g1 = img1_d(:,:,2); g1_masked = g1(idx);
    b1 = img1_d(:,:,3); b1_masked = b1(idx);
    
    % 图二
    r2 = img2_d(:,:,1); r2_masked = r2(idx);
    g2 = img2_d(:,:,2); g2_masked = g2(idx);
    b2 = img2_d(:,:,3); b2_masked = b2(idx);
    mse_r = mean((r1_masked - r2_masked).^2);
    mse_g = mean((g1_masked - g2_masked).^2);
    mse_b = mean((b1_masked - b2_masked).^2);
    mse_12 = (mse_r + mse_g + mse_b) / 3;
    psnr_12 = 10 * log10(1 / mse_12);
    
    % 图三
    r3 = img3_d(:,:,1); r3_masked = r3(idx);
    g3 = img3_d(:,:,2); g3_masked = g3(idx);
    b3 = img3_d(:,:,3); b3_masked = b3(idx);
    mse_r = mean((r1_masked - r3_masked).^2);
    mse_g = mean((g1_masked - g3_masked).^2);
    mse_b = mean((b1_masked - b3_masked).^2);
    mse_13 = (mse_r + mse_g + mse_b) / 3;
    psnr_13 = 10 * log10(1 / mse_13);
    
    % 图四
    r4 = img4_d(:,:,1); r4_masked = r4(idx);
    g4 = img4_d(:,:,2); g4_masked = g4(idx);
    b4 = img4_d(:,:,3); b4_masked = b4(idx);
    mse_r = mean((r1_masked - r4_masked).^2);
    mse_g = mean((g1_masked - g4_masked).^2);
    mse_b = mean((b1_masked - b4_masked).^2);
    mse_14 = (mse_r + mse_g + mse_b) / 3;
    psnr_14 = 10 * log10(1 / mse_14);
    
    % 存储当前区域的结果
    regionMetrics(k, :) = [ssim_12, ssim_13, ssim_14, psnr_12, psnr_13, psnr_14];
end

%% 8. 计算最终平均结果
avgSSIM_12 = mean(regionMetrics(:,1));
avgSSIM_13 = mean(regionMetrics(:,2));
avgSSIM_14 = mean(regionMetrics(:,3));
avgPSNR_12 = mean(regionMetrics(:,4));
avgPSNR_13 = mean(regionMetrics(:,5));
avgPSNR_14 = mean(regionMetrics(:,6));

%% 9. 输出结果
fprintf('\n========== 平均结果（所有正方形区域） ==========\n');
fprintf('图一 vs 图二：平均 SSIM = %.4f, 平均 PSNR = %.2f dB\n', avgSSIM_12, avgPSNR_12);
fprintf('图一 vs 图三：平均 SSIM = %.4f, 平均 PSNR = %.2f dB\n', avgSSIM_13, avgPSNR_13);
fprintf('图一 vs 图四：平均 SSIM = %.4f, 平均 PSNR = %.2f dB\n', avgSSIM_14, avgPSNR_14);

% 可选：显示各区域详细结果（调试用）
% disp('各区域详细结果（每行：SSIM12, SSIM13, SSIM14, PSNR12, PSNR13, PSNR14）:');
% disp(regionMetrics);

%% 10. 清理临时变量（可选）
clear idx r1 r2 r3 r4 g1 g2 g3 g4 b1 b2 b3 b4 ...
      mse_r mse_g mse_b mse_12 mse_13 mse_14 ...
      ssim_12 ssim_13 ssim_14 psnr_12 psnr_13 psnr_14 ...
      currentMask L sz1 sz2 sz3 sz4 szM threshold ...
      file1 file2 file3 file4 fileM path1 path2 path3 path4 pathM