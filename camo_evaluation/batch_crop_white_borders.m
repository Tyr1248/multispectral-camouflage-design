% batch_crop_white_borders_with_top_crop.m
% Function: batch-remove white borders around images, plus crop an additional
% top region of a given height (used to remove titles)
% Notes: iterate over all images in the specified folder; for each image:
%       1. automatically detect and crop the white border on all four sides;
%       2. then crop the given number of pixel rows from the top (set by extra_crop_top).
%       Results are saved in the same folder ('_cropped' appended to the filename).

%% Parameter settings
input_folder  = 'white_edge_images';   % input image folder (change to the actual path)
% Output goes to the same folder (no separate output_folder needed)
white_threshold = 240/255;        % white threshold for uint8 images (240/255 ≈ 0.941)
extra_crop_top = 50;              % number of extra top pixel rows to crop (adjust to the title height)
extensions = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'};

%% Collect all image files
files = [];
for i = 1:length(extensions)
    files = [files; dir(fullfile(input_folder, extensions{i}))];
end

if isempty(files)
    error('在指定文件夹中未找到任何支持的图像文件。');
end

%% Batch processing
for k = 1:length(files)
    filename = files(k).name;
    filepath = fullfile(input_folder, filename);
    fprintf('正在处理 (%d/%d): %s\n', k, length(files), filename);
    
    % Read the image
    img = imread(filepath);
    
    % Step 1: crop the white border on all four sides
    img_no_border = cropWhiteBorder(img, white_threshold);

    % Step 2: crop the extra top rows (if the image is tall enough)
    if extra_crop_top > 0
        h = size(img_no_border, 1);
        if extra_crop_top >= h
            warning('extra_crop_top 大于或等于图像高度，图像将被完全裁剪。请检查参数。');
            img_cropped = [];  % empty image; handle as needed
        else
            img_cropped = img_no_border(extra_crop_top+1:end, :, :);
        end
    else
        img_cropped = img_no_border;
    end
    
    % Save the result (if non-empty after cropping)
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
% Sub-function: crop the white border of a single image (as above)
%% ------------------------------------------------------------------------
function img_cropped = cropWhiteBorder(img, threshold)
    % Normalize to [0,1]
    if isa(img, 'uint8')
        img_norm = double(img) / 255;
    elseif isa(img, 'uint16')
        img_norm = double(img) / 65535;
    else
        img_norm = double(img);
    end
    
    % Identify white pixels
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
    
    % Positions of non-white pixels
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