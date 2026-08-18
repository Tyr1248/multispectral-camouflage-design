% Replace the 256x256 square regions marked by a mask with a solid color
% Function: interactively select the original image and a mask image, then
% replace the white connected regions of area 256*256 in the mask with a
% user-specified solid color
% The result is saved as: <original filename>_singlecolor.<extension>

clear; clc; close all;

%% 1. Interactively select the original image and the mask image
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

%% 2. Read the images
img_orig = imread(orig_full);   % original image
mask_img = imread(mask_full);   % mask image

% Check that the sizes match
if size(img_orig, 1) ~= size(mask_img, 1) || size(img_orig, 2) ~= size(mask_img, 2)
    error('原图像与掩膜图像的分辨率不一致，请检查。');
end

%% 3. Convert the mask to a binary logical array
% If the mask is a color image, convert it to grayscale first, then binarize
if size(mask_img, 3) == 3
    mask_gray = rgb2gray(mask_img);
else
    mask_gray = mask_img;
end
% Assume white regions in the mask (close to 1) are the regions of interest;
% binarize with a threshold of 0.5
mask_bw = imbinarize(mask_gray, 0.5);

%% 4. Find connected regions in the mask and keep those with area 256x256
cc = bwconncomp(mask_bw);          % compute connected components
stats = regionprops(cc, 'Area', 'PixelIdxList', 'BoundingBox');

% Target area: 256*256
target_area = 256 * 256;
valid_idx = find([stats.Area] == target_area);

if isempty(valid_idx)
    warning('未找到面积为 %d 的正方形区域，将处理所有连通区域。', target_area);
    valid_idx = 1:length(stats);   % if none qualify, process all connected regions
else
    fprintf('找到 %d 个面积为 256x256 的区域。\n', length(valid_idx));
end

%% 5. Interactively enter the fill color (RGB values, 0-255)
prompt = {'红色分量 (0-255):', '绿色分量 (0-255):', '蓝色分量 (0-255):'};
dlgtitle = '设置填充颜色';
dims = [1 35];
definput = {'255', '0', '0'};   % default: red
answer = inputdlg(prompt, dlgtitle, dims, definput);

if isempty(answer)
    disp('用户取消颜色输入，程序退出。');
    return;
end

% Convert the input strings to numbers and clamp them to the 0-255 range
r = max(0, min(255, round(str2double(answer{1}))));
g = max(0, min(255, round(str2double(answer{2}))));
b = max(0, min(255, round(str2double(answer{3}))));
color_rgb = [r, g, b];

% Convert the color values to the data type of the original image
if isa(img_orig, 'uint8')
    color = uint8(color_rgb);
elseif isa(img_orig, 'uint16')
    % For uint16 the usual range is 0-65535; simple mapping: multiply by 257 (255*257≈65535)
    color = uint16(color_rgb) * 257;
elseif isa(img_orig, 'double')
    % For double images, pixel values are usually in the 0-1 range
    color = double(color_rgb) / 255;
else
    % Other types (e.g. int16, logical) are uncommon; use the values as-is
    color = color_rgb;
end

%% 6. Replace the corresponding regions of the original image with the solid color
img_out = img_orig;   % copy the original image for modification

% Check whether the original image is color or grayscale
if size(img_orig, 3) == 3
    % Color image: process the R, G, B channels separately
    for i = 1:length(valid_idx)
        idx = stats(valid_idx(i)).PixelIdxList;   % linear indices of the current region
        for c = 1:3
            channel = img_out(:,:,c);
            channel(idx) = color(c);
            img_out(:,:,c) = channel;
        end
    end
else
    % Grayscale image: use the mean of the color components as the fill gray value
    if isa(img_orig, 'double')
        gray_val = double(mean(color_rgb)) / 255;
    elseif isa(img_orig, 'uint8')
        gray_val = uint8(round(mean(color_rgb)));
    elseif isa(img_orig, 'uint16')
        gray_val = uint16(round(mean(color_rgb) * 257));
    else
        gray_val = mean(color_rgb);   % other types: take the mean directly
    end
    for i = 1:length(valid_idx)
        idx = stats(valid_idx(i)).PixelIdxList;
        img_out(idx) = gray_val;
    end
end

%% 7. Save the result image
[~, name, ext] = fileparts(orig_file);
new_filename = [name, '_singlecolor', ext];
new_fullpath = fullfile(orig_path, new_filename);

try
    imwrite(img_out, new_fullpath);
    fprintf('结果图像已保存至：%s\n', new_fullpath);
catch ME
    error('保存图像失败：%s', ME.message);
end

%% 8. Display the original image and the result side by side
figure('Name', '替换效果对比');
subplot(1,2,1);
imshow(img_orig);
title('原图像');
subplot(1,2,2);
imshow(img_out);
title('替换后图像');

disp('程序运行完成。');