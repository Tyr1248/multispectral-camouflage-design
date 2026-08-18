addRedBoxToImage('ourcamo_Saliency\ourcamo_Itti_map_overlayed.png', 'samplepics/mask.png', 7);
function addRedBoxToImage(input_image_path, mask_image_path, lineSize)
    % Complete red-box drawing function - supports multiple mask regions
    % Inputs:
    %   input_image_path: path to the original image
    %   mask_image_path: path to the mask image
    %   lineSize: red box line width (pixels)

    % Parameter validation and default values
    if nargin < 3
        lineSize = 3; % default line width
    end

    % Read the original image and the mask
    try
        img = imread(input_image_path);
        mask = imread(mask_image_path);
    catch ME
        error('无法读取图像文件: %s', ME.message);
    end
    
    % Ensure the mask is a binary image
    if ~islogical(mask)
        if size(mask, 3) > 1
            mask = rgb2gray(mask);
        end
        mask = mask > 0;
    end

    % Check that the mask contains valid regions
    if ~any(mask(:))
        error('掩膜中未找到任何有效区域。');
    end

    % Find the bounding boxes of all mask regions
    stats = regionprops(mask, 'BoundingBox', 'Area');
    if isempty(stats)
        error('在掩膜中未找到任何区域。');
    end

    % Display the number of regions found
    fprintf('找到 %d 个掩膜区域\n', length(stats));

    % Copy the original image for drawing
    img_with_rect = img;
    if size(img_with_rect, 3) == 1
        img_with_rect = cat(3, img_with_rect, img_with_rect, img_with_rect);
    end

    % Draw a red box for each region
    for i = 1:length(stats)
        bbox = stats(i).BoundingBox; % [x y w h]
        area = stats(i).Area;

        % Extract coordinates and round to integers
        x1 = round(bbox(1));
        y1 = round(bbox(2));
        w = round(bbox(3));
        h = round(bbox(4));

        fprintf('处理区域 %d: 位置 [%d, %d], 大小 [%d, %d], 面积 %d\n', ...
                i, x1, y1, w, h, area);

        % Draw the red box
        img_with_rect = drawRect(img_with_rect, [x1, y1], [w, h], lineSize);
    end

    % Build the output filename: input image name + "Marked"
    [path, name, ext] = fileparts(input_image_path);
    output_name = fullfile(path, [name, 'Marked', ext]);

    % Save the image
    imwrite(img_with_rect, output_name);
    fprintf('已成功保存标记图像: %s\n', output_name);

    % Display the processing result
    figure('Name', '红框标记结果', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 400]);
    subplot(1, 3, 1);
    imshow(img);
    title('原图');
    
    subplot(1, 3, 2);
    imshow(mask);
    title('掩膜');
    
    subplot(1, 3, 3);
    imshow(img_with_rect);
    title(sprintf('添加红框后 (%d个区域)', length(stats)));
end

function dest = drawRect(src, pt, wSize, lineSize)
    % Sub-function that draws a rectangle on an image
    % Inputs:
    %   src: source image
    %   pt: top-left corner coordinates [x1, y1]
    %   wSize: box size [wx, wy]
    %   lineSize: line width
    % Output:
    %   dest: image with the rectangle drawn
    
    % Get image dimensions
    [yA, xA, z] = size(src);
    x1 = pt(1);
    y1 = pt(2);
    wx = wSize(1);
    wy = wSize(2);
    
    % Ensure the destination image has 3 channels
    if z == 1
        dest = cat(3, src, src, src);
    else
        dest = src;
    end

    % Define red
    red_color = [255, 0, 0];

    % Draw the rectangle (four edges)
    for dl = 0:(lineSize-1)
        % Top line (from x1 to x1+wx, at row y1-dl)
        row_up = y1 - dl;
        if row_up >= 1 && row_up <= yA
            cols_up = max(1, x1):min(xA, x1+wx);
            if ~isempty(cols_up)
                dest(row_up, cols_up, 1) = red_color(1); % R channel
                dest(row_up, cols_up, 2) = red_color(2); % G channel
                dest(row_up, cols_up, 3) = red_color(3); % B channel
            end
        end

        % Bottom line (from x1 to x1+wx, at row y1+wy+dl)
        row_down = y1 + wy + dl;
        if row_down >= 1 && row_down <= yA
            cols_down = max(1, x1):min(xA, x1+wx);
            if ~isempty(cols_down)
                dest(row_down, cols_down, 1) = red_color(1);
                dest(row_down, cols_down, 2) = red_color(2);
                dest(row_down, cols_down, 3) = red_color(3);
            end
        end

        % Left line (from y1 to y1+wy, at column x1-dl)
        col_left = x1 - dl;
        if col_left >= 1 && col_left <= xA
            rows_left = max(1, y1):min(yA, y1+wy);
            if ~isempty(rows_left)
                dest(rows_left, col_left, 1) = red_color(1);
                dest(rows_left, col_left, 2) = red_color(2);
                dest(rows_left, col_left, 3) = red_color(3);
            end
        end

        % Right line (from y1 to y1+wy, at column x1+wx+dl)
        col_right = x1 + wx + dl;
        if col_right >= 1 && col_right <= xA
            rows_right = max(1, y1):min(yA, y1+wy);
            if ~isempty(rows_right)
                dest(rows_right, col_right, 1) = red_color(1);
                dest(rows_right, col_right, 2) = red_color(2);
                dest(rows_right, col_right, 3) = red_color(3);
            end
        end
    end
end