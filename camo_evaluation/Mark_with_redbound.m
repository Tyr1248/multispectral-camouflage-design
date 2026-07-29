addRedBoxToImage('ourcamo_Saliency\ourcamo_Itti_map_overlayed.png', 'samplepics/mask.png', 7);
function addRedBoxToImage(input_image_path, mask_image_path, lineSize)
    % 完整的红框添加函数 - 支持多个掩膜区域
    % 输入：
    %   input_image_path: 原图文件路径
    %   mask_image_path: 掩膜文件路径  
    %   lineSize: 红框线宽（像素）
    
    % 参数验证和默认值设置
    if nargin < 3
        lineSize = 3; % 默认线宽
    end
    
    % 读取原图和掩膜
    try
        img = imread(input_image_path);
        mask = imread(mask_image_path);
    catch ME
        error('无法读取图像文件: %s', ME.message);
    end
    
    % 确保掩膜是二值图像
    if ~islogical(mask)
        if size(mask, 3) > 1
            mask = rgb2gray(mask);
        end
        mask = mask > 0;
    end
    
    % 检查掩膜是否包含有效区域
    if ~any(mask(:))
        error('掩膜中未找到任何有效区域。');
    end
    
    % 找到所有掩膜区域的边界框
    stats = regionprops(mask, 'BoundingBox', 'Area');
    if isempty(stats)
        error('在掩膜中未找到任何区域。');
    end
    
    % 显示找到的区域数量
    fprintf('找到 %d 个掩膜区域\n', length(stats));
    
    % 复制原图用于绘制
    img_with_rect = img;
    if size(img_with_rect, 3) == 1
        img_with_rect = cat(3, img_with_rect, img_with_rect, img_with_rect);
    end
    
    % 为每个区域绘制红框
    for i = 1:length(stats)
        bbox = stats(i).BoundingBox; % [x y w h]
        area = stats(i).Area;
        
        % 提取坐标并确保为整数
        x1 = round(bbox(1));
        y1 = round(bbox(2));
        w = round(bbox(3));
        h = round(bbox(4));
        
        fprintf('处理区域 %d: 位置 [%d, %d], 大小 [%d, %d], 面积 %d\n', ...
                i, x1, y1, w, h, area);
        
        % 绘制红框
        img_with_rect = drawRect(img_with_rect, [x1, y1], [w, h], lineSize);
    end
    
    % 生成输出文件名：输入图像名 + "Marked"
    [path, name, ext] = fileparts(input_image_path);
    output_name = fullfile(path, [name, 'Marked', ext]);
    
    % 保存图像
    imwrite(img_with_rect, output_name);
    fprintf('已成功保存标记图像: %s\n', output_name);
    
    % 显示处理结果
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
    % 在图像上绘制矩形框的子函数
    % 输入：
    %   src: 原始图像
    %   pt: 左上角坐标 [x1, y1]  
    %   wSize: 框的大小 [wx, wy]
    %   lineSize: 线的宽度
    % 输出：
    %   dest: 绘制了矩形框的图像
    
    % 获取图像尺寸
    [yA, xA, z] = size(src);
    x1 = pt(1);
    y1 = pt(2);
    wx = wSize(1);
    wy = wSize(2);
    
    % 确保目标图像是3通道
    if z == 1
        dest = cat(3, src, src, src);
    else
        dest = src;
    end
    
    % 定义红色
    red_color = [255, 0, 0];
    
    % 绘制矩形框（四个边）
    for dl = 0:(lineSize-1)
        % 上方线条（从x1到x1+wx，y1-dl位置）
        row_up = y1 - dl;
        if row_up >= 1 && row_up <= yA
            cols_up = max(1, x1):min(xA, x1+wx);
            if ~isempty(cols_up)
                dest(row_up, cols_up, 1) = red_color(1); % R通道
                dest(row_up, cols_up, 2) = red_color(2); % G通道  
                dest(row_up, cols_up, 3) = red_color(3); % B通道
            end
        end
        
        % 下方线条（从x1到x1+wx，y1+wy+dl位置）
        row_down = y1 + wy + dl;
        if row_down >= 1 && row_down <= yA
            cols_down = max(1, x1):min(xA, x1+wx);
            if ~isempty(cols_down)
                dest(row_down, cols_down, 1) = red_color(1);
                dest(row_down, cols_down, 2) = red_color(2);
                dest(row_down, cols_down, 3) = red_color(3);
            end
        end
        
        % 左方线条（从y1到y1+wy，x1-dl位置）
        col_left = x1 - dl;
        if col_left >= 1 && col_left <= xA
            rows_left = max(1, y1):min(yA, y1+wy);
            if ~isempty(rows_left)
                dest(rows_left, col_left, 1) = red_color(1);
                dest(rows_left, col_left, 2) = red_color(2);
                dest(rows_left, col_left, 3) = red_color(3);
            end
        end
        
        % 右方线条（从y1到y1+wy，x1+wx+dl位置）
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