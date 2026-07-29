function [CCE, SL, SC, ST] = evaluateCamouflageEffect(img, mask)
    % ========== 自适应下采样（以目标区域为基准）==========
    max_target_area = 64*64;  % 可接受的最大目标像素数（如 100x100）
    min_target_area = 256;    % 最小目标像素数（防过度缩小）

    current_area = nnz(mask);
    
    if current_area > max_target_area
        scale = sqrt(double(max_target_area) / double(current_area));
        scale = max(scale, 0.1); % 防止缩得太小
    elseif current_area < min_target_area
        scale = sqrt(double(min_target_area) / double(current_area));
        scale = min(scale, 2.0); % 防止放大太多
    else
        scale = 1.0;
    end

    % 应用缩放（仅当 scale ≠ 1）
    if abs(scale - 1.0) > 1e-3
        new_size = round(size(img,1:2) * scale);
        % 避免尺寸为0
        new_size = max(new_size, [1, 1]);
        
        img = imresize(img, new_size, 'bilinear');
        mask = imresize(mask, new_size, 'nearest') > 0.5;  % 保持二值性
    end

    % ========== 后续处理（不变）==========
    sigma = 30;          % 空间权重参数（注意：现在基于新尺寸）
    alpha_sigmoid = 0.01;

    wL = 0.501;
    wC = 0.366;
    wT = 0.133;

    [target_region, target_points, bg_region, bg_points] = extractTargetAndBackground(img, mask);

    SL = computeBrightnessSimilarity(target_region, bg_region, target_points, bg_points, sigma);
    SC = computeColorSimilarity(target_region, bg_region, target_points, bg_points, sigma);
    ST = computeTextureFusionDegree(target_region, bg_region, alpha_sigmoid);

    CCE = wL * SL + wC * SC + wT * ST;
end

function [T_img, T_pts, B_img, B_pts] = extractTargetAndBackground(img, mask)
    [rows, cols, ~] = size(img);
    [y, x] = find(mask);  % 获取目标像素坐标
    
    if isempty(x) || isempty(y)
        error('目标区域为空，请检查掩膜。');
    end

    % 计算目标区域边界框
    top = min(y); bottom = max(y);
    left = min(x); right = max(x);
    
    h = bottom - top + 1;
    w = right - left + 1;
    
    % === 提取目标区域 ===
    T_img_raw = img(top:bottom, left:right, :);  % Ht x Wt x 3
    T_mask_raw = true(size(T_img_raw,1), size(T_img_raw,2));  % 全为目标
    T_pixels = reshape(T_img_raw, [], 3);  % Nt x 3
    [ty, tx] = find(T_mask_raw);
    T_pts = [tx + left - 1, ty + top - 1];  % 转回原图坐标

    % === 定义背景大区域（目标周围同尺寸八连通环）===
    bg_top = max(1, top - h);
    bg_bottom = min(rows, bottom + h);
    bg_left = max(1, left - w);
    bg_right = min(cols, right + w);

    big_region = img(bg_top:bg_bottom, bg_left:bg_right, :);  % Hb x Wb x 3
    big_H = size(big_region, 1);
    big_W = size(big_region, 2);

    % 构建 big_mask：标记其中哪些位置属于原始目标区域
    big_mask = false(big_H, big_W);
    tgt_in_big_y1 = top - bg_top + 1;
    tgt_in_big_x1 = left - bg_left + 1;
    tgt_in_big_y2 = tgt_in_big_y1 + h - 1;
    tgt_in_big_x2 = tgt_in_big_x1 + w - 1;

    % 安全检查：确保目标区域完全落在 big_region 内
    if tgt_in_big_y1 < 1 || tgt_in_big_x1 < 1 || ...
       tgt_in_big_y2 > big_H || tgt_in_big_x2 > big_W
        error('目标区域超出背景大区域范围，请检查图像边界。');
    end

    big_mask(tgt_in_big_y1:tgt_in_big_y2, tgt_in_big_x1:tgt_in_big_x2) = true;

    % === 关键修复：正确使用逻辑索引 ===
    % 将 big_region 展平为 (H*W) x 3
    big_region_2d = reshape(big_region, [], 3);          % (H*W) x 3
    big_mask_vec = big_mask(:);                         % (H*W) x 1 logical

    % 背景像素：非目标部分
    bg_mask_vec = ~big_mask_vec;
    if ~any(bg_mask_vec)
        error('背景区域为空（目标占满整个扩展区域），无法计算伪装效果。');
    end

    B_pixels = big_region_2d(bg_mask_vec, :);           % Nb x 3
    B_img = B_pixels;

    % 获取背景点在 big_region 中的 (row, col)，再转回原图坐标
    [bg_rows_big, bg_cols_big] = find(bg_mask_vec);     % 在 big_region 展平前的行列索引
    B_pts = [bg_cols_big + bg_left - 1, bg_rows_big + bg_top - 1];
    
    % 返回目标像素（也展平）
    T_img = T_pixels;
end

function SL = computeBrightnessSimilarity(T_rgb, B_rgb, T_pts, B_pts, sigma)
    % 将RGB转为CIELab，提取L通道
    T_lab = rgb2lab(T_rgb);
    B_lab = rgb2lab(B_rgb);
    L_T = T_lab(:,1);
    L_B = B_lab(:,1);
    
    % 计算空间加权欧氏距离相似度
    SL = computeSpatialWeightedEuclideanDistance(L_T, L_B, T_pts, B_pts, sigma);
end

function SC = computeColorSimilarity(T_rgb, B_rgb, T_pts, B_pts, sigma)
    % 将RGB转为HSV
    T_hsv = rgb2hsv(T_rgb / 255);
    B_hsv = rgb2hsv(B_rgb / 255);
    
    % 量化 HSV（按论文：H:12份, S:5份, V:5份）
    H_T = floor(T_hsv(:,1) * 12); H_T(H_T==12) = 11;
    S_T = floor(T_hsv(:,2) * 5);  S_T(S_T==5) = 4;
    V_T = floor(T_hsv(:,3) * 5);  V_T(V_T==5) = 4;
    C_T = 25 * H_T + 5 * S_T + V_T; % 公式7

    H_B = floor(B_hsv(:,1) * 12); H_B(H_B==12) = 11;
    S_B = floor(B_hsv(:,2) * 5);  S_B(S_B==5) = 4;
    V_B = floor(B_hsv(:,3) * 5);  V_B(V_B==5) = 4;
    C_B = 25 * H_B + 5 * S_B + V_B;

    % 计算相似度
    SC = computeSpatialWeightedEuclideanDistance(C_T, C_B, T_pts, B_pts, sigma);
end

function ST = computeTextureFusionDegree(T_rgb, B_rgb, alpha)
    % 使用Gabor滤波器提取纹理（简化版：仅用一个尺度+4方向）
    orientations = [0, pi/4, pi/2, 3*pi/4];
    lambda = 10; % 波长
    gamma = 0.5;
    psi = 0;
    sigma_g = 5;
    
    % 转灰度
    T_gray = rgb2gray(T_rgb);
    B_gray = rgb2gray(B_rgb);
    
    % 提取Gabor响应均值和方差（对所有方向合并）
    feat_T = extractGaborStats(T_gray, orientations, sigma_g, lambda, gamma, psi);
    feat_B = extractGaborStats(B_gray, orientations, sigma_g, lambda, gamma, psi);
    
    % 计算欧氏距离
    d = norm(feat_T - feat_B);
    
    % Sigmoid归一化（公式5 & 9）
    Sx = 1 / (1 + exp(-alpha * d));
    ST = 1 - 0.5 * (Sx - 0.5); % 公式9
end

function feat = extractGaborStats(img, thetas, sigma, lambda, gamma, psi)
    responses = [];
    for k = 1:length(thetas)
        theta = thetas(k);
        g = fspecial('gabor', [sigma, theta, 0, lambda, gamma, psi]);
        filtered = imfilter(double(img), g, 'symmetric');
        responses = [responses; mean(filtered(:)); var(filtered(:))];
    end
    feat = mean(responses); % 这里简化处理，直接使用均值作为特征向量
end

function S = computeSpatialWeightedEuclideanDistance(T_vec, B_vec, T_pts, B_pts, sigma)
    n = length(T_vec);
    m = length(B_vec);
    
    if m == 0 || n == 0
        S = 0; return;
    end
    
    % 初始化权重矩阵
    weights = zeros(n, m);
    
    % 计算每个目标点与背景点之间的权重
    for i = 1:n
        for j = 1:m
            dist = pdist2(T_pts(i,:), B_pts(j,:));
            weights(i,j) = exp(-(dist^2)/(2*sigma^2));
        end
    end
    
    % 归一化权重
    weights = bsxfun(@rdivide, weights, sum(weights,2));
    
    % 计算加权欧氏距离
    D = zeros(n,1);
    for i = 1:n
        D(i) = sqrt(sum(bsxfun(@times, weights(i,:), (T_vec(i)-B_vec).^2)));
    end
    
    % 计算总体相似度
    S = 1 - mean(D);
end