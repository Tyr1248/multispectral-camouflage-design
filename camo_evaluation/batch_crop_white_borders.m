% batch_crop_white_borders_with_top_crop.m
% 功能：批量去除图像外围白色边框，并额外裁剪掉指定高度的顶部区域（用于移除标题）
% 说明：遍历指定文件夹中的所有图像，对每张图像：
%       1. 自动检测并裁剪四周白边；
%       2. 再裁剪掉顶部指定的像素行（由 extra_crop_top 设定）。
%       结果保存在同一文件夹（文件名添加 '_cropped' 后缀）。

%% 参数设置
input_folder  = 'white_edge_images';   % 输入图像文件夹（请修改为实际路径）
% 输出到同一文件夹（无需单独设置 output_folder）
white_threshold = 240/255;        % 白色阈值，适用于 uint8 图像（240/255 ≈ 0.941）
extra_crop_top = 50;              % 额外裁剪的顶部像素行数（请根据标题高度调整）
extensions = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'};

%% 获取所有图像文件
files = [];
for i = 1:length(extensions)
    files = [files; dir(fullfile(input_folder, extensions{i}))];
end

if isempty(files)
    error('在指定文件夹中未找到任何支持的图像文件。');
end

%% 批量处理
for k = 1:length(files)
    filename = files(k).name;
    filepath = fullfile(input_folder, filename);
    fprintf('正在处理 (%d/%d): %s\n', k, length(files), filename);
    
    % 读取图像
    img = imread(filepath);
    
    % 第一步：裁剪四周白边
    img_no_border = cropWhiteBorder(img, white_threshold);
    
    % 第二步：额外裁剪顶部指定行数（若图像高度足够）
    if extra_crop_top > 0
        h = size(img_no_border, 1);
        if extra_crop_top >= h
            warning('extra_crop_top 大于或等于图像高度，图像将被完全裁剪。请检查参数。');
            img_cropped = [];  % 图像为空，可自行处理
        else
            img_cropped = img_no_border(extra_crop_top+1:end, :, :);
        end
    else
        img_cropped = img_no_border;
    end
    
    % 保存结果（若裁剪后非空）
    if ~isempty(img_cropped)
        [~, name, ext] = fileparts(filename);
        output_filename = [name '_cropped' ext];
        output_path = fullfile(input_folder, output_filename);
        imwrite(img_cropped, output_path);
    else
        warning('图像 %s 裁剪后为空，未保存。', filename);
    end
end

fprintf('批量处理完成！结果已保存至: %s\n', input_folder);

%% ------------------------------------------------------------------------
% 子函数：裁剪单张图像的白色边框（同前）
%% ------------------------------------------------------------------------
function img_cropped = cropWhiteBorder(img, threshold)
    % 归一化到 [0,1]
    if isa(img, 'uint8')
        img_norm = double(img) / 255;
    elseif isa(img, 'uint16')
        img_norm = double(img) / 65535;
    else
        img_norm = double(img);
    end
    
    % 判断白色像素
    if size(img_norm, 3) == 1
        is_white = img_norm >= threshold;
    else
        if isscalar(threshold)
            is_white = all(img_norm >= threshold, 3);
        else
            is_white = true(size(img_norm,1), size(img_norm,2));
            for c = 1:3
                is_white = is_white & (img_norm(:,:,c) >= threshold(c));
            end
        end
    end
    
    % 非白色像素位置
    [rows, cols] = find(~is_white);
    
    if isempty(rows) || isempty(cols)
        warning('图像全白或未检测到非白色像素，返回原图。');
        img_cropped = img;
        return;
    end
    
    top    = min(rows);
    bottom = max(rows);
    left   = min(cols);
    right  = max(cols);
    
    img_cropped = img(top:bottom, left:right, :);
end