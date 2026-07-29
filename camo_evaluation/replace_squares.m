% 替换图像中被掩膜标注的256x256正方形区域为纯色
% 功能：交互选择原图像和掩膜图像，将掩膜中面积为256*256的白色连通区域替换为用户指定的纯色
% 结果图像保存为：原文件名_singlecolor.扩展名

clear; clc; close all;

%% 1. 交互选择原图像和掩膜图像
[orig_file, orig_path] = uigetfile( ...
    {'*.jpg;*.png;*.bmp;*.tif;*.jpeg', '图像文件 (*.jpg,*.png,*.bmp,*.tif,*.jpeg)'; ...
     '*.*', '所有文件 (*.*)'}, ...
    '请选择原图像');
if isequal(orig_file, 0)
    disp('用户取消选择原图像，程序退出。');
    return;
end

[mask_file, mask_path] = uigetfile( ...
    {'*.jpg;*.png;*.bmp;*.tif;*.jpeg', '图像文件 (*.jpg,*.png,*.bmp,*.tif,*.jpeg)'; ...
     '*.*', '所有文件 (*.*)'}, ...
    '请选择掩膜图像（二值图，白色区域为待替换区域）');
if isequal(mask_file, 0)
    disp('用户取消选择掩膜图像，程序退出。');
    return;
end

orig_full = fullfile(orig_path, orig_file);
mask_full = fullfile(mask_path, mask_file);

%% 2. 读取图像
img_orig = imread(orig_full);   % 原图像
mask_img = imread(mask_full);   % 掩膜图像

% 检查尺寸一致性
if size(img_orig, 1) ~= size(mask_img, 1) || size(img_orig, 2) ~= size(mask_img, 2)
    error('原图像与掩膜图像的分辨率不一致，请检查。');
end

%% 3. 将掩膜转换为二值逻辑数组
% 如果掩膜是彩色图，转为灰度；然后二值化
if size(mask_img, 3) == 3
    mask_gray = rgb2gray(mask_img);
else
    mask_gray = mask_img;
end
% 假设掩膜中白色区域（接近1）为感兴趣区域，使用阈值0.5进行二值化
mask_bw = imbinarize(mask_gray, 0.5);

%% 4. 查找掩膜中的连通区域，并筛选面积为256x256的区域
cc = bwconncomp(mask_bw);          % 计算连通组件
stats = regionprops(cc, 'Area', 'PixelIdxList', 'BoundingBox');

% 目标面积：256*256
target_area = 256 * 256;
valid_idx = find([stats.Area] == target_area);

if isempty(valid_idx)
    warning('未找到面积为 %d 的正方形区域，将处理所有连通区域。', target_area);
    valid_idx = 1:length(stats);   % 如果没有符合条件的，则处理所有连通区域
else
    fprintf('找到 %d 个面积为 256x256 的区域。\n', length(valid_idx));
end

%% 5. 交互输入要填充的纯色（RGB值，0-255）
prompt = {'红色分量 (0-255):', '绿色分量 (0-255):', '蓝色分量 (0-255):'};
dlgtitle = '设置填充颜色';
dims = [1 35];
definput = {'255', '0', '0'};   % 默认红色
answer = inputdlg(prompt, dlgtitle, dims, definput);

if isempty(answer)
    disp('用户取消颜色输入，程序退出。');
    return;
end

% 将输入字符串转换为数值，并限制在0-255范围内
r = max(0, min(255, round(str2double(answer{1}))));
g = max(0, min(255, round(str2double(answer{2}))));
b = max(0, min(255, round(str2double(answer{3}))));
color_rgb = [r, g, b];

% 根据原图像的数据类型，将颜色值转换为相应类型
if isa(img_orig, 'uint8')
    color = uint8(color_rgb);
elseif isa(img_orig, 'uint16')
    % 对于uint16，通常范围是0-65535，简单映射：乘以257（255*257≈65535）
    color = uint16(color_rgb) * 257;
elseif isa(img_orig, 'double')
    % 对于double类型图像，像素值通常在0-1之间
    color = double(color_rgb) / 255;
else
    % 其他类型（如int16、logical等）可能不常见，直接使用原值
    color = color_rgb;
end

%% 6. 将原图中对应区域替换为纯色
img_out = img_orig;   % 复制原图，后续修改

% 判断原图像是彩色还是灰度
if size(img_orig, 3) == 3
    % 彩色图像：分别处理R,G,B三个通道
    for i = 1:length(valid_idx)
        idx = stats(valid_idx(i)).PixelIdxList;   % 当前区域的线性索引
        for c = 1:3
            channel = img_out(:,:,c);
            channel(idx) = color(c);
            img_out(:,:,c) = channel;
        end
    end
else
    % 灰度图像：使用颜色分量的平均值作为填充灰度值
    if isa(img_orig, 'double')
        gray_val = double(mean(color_rgb)) / 255;
    elseif isa(img_orig, 'uint8')
        gray_val = uint8(round(mean(color_rgb)));
    elseif isa(img_orig, 'uint16')
        gray_val = uint16(round(mean(color_rgb) * 257));
    else
        gray_val = mean(color_rgb);   % 其他类型直接取均值
    end
    for i = 1:length(valid_idx)
        idx = stats(valid_idx(i)).PixelIdxList;
        img_out(idx) = gray_val;
    end
end

%% 7. 保存结果图像
[~, name, ext] = fileparts(orig_file);
new_filename = [name, '_singlecolor', ext];
new_fullpath = fullfile(orig_path, new_filename);

try
    imwrite(img_out, new_fullpath);
    fprintf('结果图像已保存至：%s\n', new_fullpath);
catch ME
    error('保存图像失败：%s', ME.message);
end

%% 8. 显示原图和结果对比
figure('Name', '替换效果对比');
subplot(1,2,1);
imshow(img_orig);
title('原图像');
subplot(1,2,2);
imshow(img_out);
title('替换后图像');

disp('程序运行完成。');