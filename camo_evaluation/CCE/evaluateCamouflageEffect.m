function [CCE, SL, SC, ST] = evaluateCamouflageEffect(img, mask)
    % ========== Adaptive downsampling (relative to the target region) ==========
    max_target_area = 64*64;  % maximum acceptable number of target pixels (e.g. 100x100)
    min_target_area = 256;    % minimum number of target pixels (prevents excessive shrinking)

    current_area = nnz(mask);
    
    if current_area > max_target_area
        scale = sqrt(double(max_target_area) / double(current_area));
        scale = max(scale, 0.1); % avoid shrinking too much
    elseif current_area < min_target_area
        scale = sqrt(double(min_target_area) / double(current_area));
        scale = min(scale, 2.0); % avoid enlarging too much
    else
        scale = 1.0;
    end

    % Apply the scaling (only when scale ≠ 1)
    if abs(scale - 1.0) > 1e-3
        new_size = round(size(img,1:2) * scale);
        % avoid a zero size
        new_size = max(new_size, [1, 1]);

        img = imresize(img, new_size, 'bilinear');
        mask = imresize(mask, new_size, 'nearest') > 0.5;  % keep it binary
    end

    % ========== Subsequent processing (unchanged) ==========
    sigma = 30;          % spatial weight parameter (note: now based on the new size)
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
    [y, x] = find(mask);  % get the target pixel coordinates
    
    if isempty(x) || isempty(y)
        error('目标区域为空，请检查掩膜。');
    end

    % Compute the bounding box of the target region
    top = min(y); bottom = max(y);
    left = min(x); right = max(x);
    
    h = bottom - top + 1;
    w = right - left + 1;
    
    % === Extract the target region ===
    T_img_raw = img(top:bottom, left:right, :);  % Ht x Wt x 3
    T_mask_raw = true(size(T_img_raw,1), size(T_img_raw,2));  % all target
    T_pixels = reshape(T_img_raw, [], 3);  % Nt x 3
    [ty, tx] = find(T_mask_raw);
    T_pts = [tx + left - 1, ty + top - 1];  % back to original image coordinates

    % === Define the large background region (an 8-connected ring of the same size around the target) ===
    bg_top = max(1, top - h);
    bg_bottom = min(rows, bottom + h);
    bg_left = max(1, left - w);
    bg_right = min(cols, right + w);

    big_region = img(bg_top:bg_bottom, bg_left:bg_right, :);  % Hb x Wb x 3
    big_H = size(big_region, 1);
    big_W = size(big_region, 2);

    % Build big_mask: mark which positions belong to the original target region
    big_mask = false(big_H, big_W);
    tgt_in_big_y1 = top - bg_top + 1;
    tgt_in_big_x1 = left - bg_left + 1;
    tgt_in_big_y2 = tgt_in_big_y1 + h - 1;
    tgt_in_big_x2 = tgt_in_big_x1 + w - 1;

    % Safety check: ensure the target region lies fully inside big_region
    if tgt_in_big_y1 < 1 || tgt_in_big_x1 < 1 || ...
       tgt_in_big_y2 > big_H || tgt_in_big_x2 > big_W
        error('目标区域超出背景大区域范围，请检查图像边界。');
    end

    big_mask(tgt_in_big_y1:tgt_in_big_y2, tgt_in_big_x1:tgt_in_big_x2) = true;

    % === Key fix: use logical indexing correctly ===
    % Flatten big_region to (H*W) x 3
    big_region_2d = reshape(big_region, [], 3);          % (H*W) x 3
    big_mask_vec = big_mask(:);                         % (H*W) x 1 logical

    % Background pixels: the non-target part
    bg_mask_vec = ~big_mask_vec;
    if ~any(bg_mask_vec)
        error('背景区域为空（目标占满整个扩展区域），无法计算伪装效果。');
    end

    B_pixels = big_region_2d(bg_mask_vec, :);           % Nb x 3
    B_img = B_pixels;

    % Get the (row, col) of background points within big_region, then convert
    % back to original image coordinates
    [bg_rows_big, bg_cols_big] = find(bg_mask_vec);     % row/col indices before flattening big_region
    B_pts = [bg_cols_big + bg_left - 1, bg_rows_big + bg_top - 1];

    % Return the target pixels (also flattened)
    T_img = T_pixels;
end

function SL = computeBrightnessSimilarity(T_rgb, B_rgb, T_pts, B_pts, sigma)
    % Convert RGB to CIELab and extract the L channel
    T_lab = rgb2lab(T_rgb);
    B_lab = rgb2lab(B_rgb);
    L_T = T_lab(:,1);
    L_B = B_lab(:,1);

    % Compute the spatially weighted Euclidean-distance similarity
    SL = computeSpatialWeightedEuclideanDistance(L_T, L_B, T_pts, B_pts, sigma);
end

function SC = computeColorSimilarity(T_rgb, B_rgb, T_pts, B_pts, sigma)
    % Convert RGB to HSV
    T_hsv = rgb2hsv(T_rgb / 255);
    B_hsv = rgb2hsv(B_rgb / 255);

    % Quantize HSV (per the paper: H: 12 bins, S: 5 bins, V: 5 bins)
    H_T = floor(T_hsv(:,1) * 12); H_T(H_T==12) = 11;
    S_T = floor(T_hsv(:,2) * 5);  S_T(S_T==5) = 4;
    V_T = floor(T_hsv(:,3) * 5);  V_T(V_T==5) = 4;
    C_T = 25 * H_T + 5 * S_T + V_T; % Eq. 7

    H_B = floor(B_hsv(:,1) * 12); H_B(H_B==12) = 11;
    S_B = floor(B_hsv(:,2) * 5);  S_B(S_B==5) = 4;
    V_B = floor(B_hsv(:,3) * 5);  V_B(V_B==5) = 4;
    C_B = 25 * H_B + 5 * S_B + V_B;

    % Compute the similarity
    SC = computeSpatialWeightedEuclideanDistance(C_T, C_B, T_pts, B_pts, sigma);
end

function ST = computeTextureFusionDegree(T_rgb, B_rgb, alpha)
    % Extract texture with Gabor filters (simplified: one scale + 4 orientations)
    orientations = [0, pi/4, pi/2, 3*pi/4];
    lambda = 10; % wavelength
    gamma = 0.5;
    psi = 0;
    sigma_g = 5;

    % Convert to grayscale
    T_gray = rgb2gray(T_rgb);
    B_gray = rgb2gray(B_rgb);

    % Extract the mean and variance of the Gabor responses (merged over all orientations)
    feat_T = extractGaborStats(T_gray, orientations, sigma_g, lambda, gamma, psi);
    feat_B = extractGaborStats(B_gray, orientations, sigma_g, lambda, gamma, psi);

    % Compute the Euclidean distance
    d = norm(feat_T - feat_B);

    % Sigmoid normalization (Eq. 5 & 9)
    Sx = 1 / (1 + exp(-alpha * d));
    ST = 1 - 0.5 * (Sx - 0.5); % Eq. 9
end

function feat = extractGaborStats(img, thetas, sigma, lambda, gamma, psi)
    responses = [];
    for k = 1:length(thetas)
        theta = thetas(k);
        g = fspecial('gabor', [sigma, theta, 0, lambda, gamma, psi]);
        filtered = imfilter(double(img), g, 'symmetric');
        responses = [responses; mean(filtered(:)); var(filtered(:))];
    end
    feat = mean(responses); % simplified: use the mean directly as the feature vector
end

function S = computeSpatialWeightedEuclideanDistance(T_vec, B_vec, T_pts, B_pts, sigma)
    n = length(T_vec);
    m = length(B_vec);

    if m == 0 || n == 0
        S = 0; return;
    end

    % Initialize the weight matrix
    weights = zeros(n, m);

    % Compute the weight between each target point and background point
    for i = 1:n
        for j = 1:m
            dist = pdist2(T_pts(i,:), B_pts(j,:));
            weights(i,j) = exp(-(dist^2)/(2*sigma^2));
        end
    end

    % Normalize the weights
    weights = bsxfun(@rdivide, weights, sum(weights,2));

    % Compute the weighted Euclidean distance
    D = zeros(n,1);
    for i = 1:n
        D(i) = sqrt(sum(bsxfun(@times, weights(i,:), (T_vec(i)-B_vec).^2)));
    end

    % Compute the overall similarity
    S = 1 - mean(D);
end