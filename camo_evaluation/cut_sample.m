% Read the image
img = imread("E:\ProjectX\Test_data_env_fig\Summer_camo.png");  % replace with your image path

% Get image dimensions
[h, w, c] = size(img);

% 1. Extract the central 512x512 region
% Compute the center point
center_h = floor(h/2);
center_w = floor(w/2);

% Compute the start and end indices
start_h = center_h - 255;  % 512/2 = 256, but the start index is 1-based
end_h = center_h + 256;    % covers 256 pixels
start_w = center_w - 255;
end_w = center_w + 256;

% Extract the central region
center_512 = zeros(512, 512, c, class(img));
for i = 1:512
    for j = 1:512
        for k = 1:c
            center_512(i, j, k) = img(start_h + i - 1, start_w + j - 1, k);
        end
    end
end

% 2. Downsample to 256x256 (average pooling)
result_256 = zeros(256, 256, c, class(img));

% Average each 2x2 block
for i = 1:256
    for j = 1:256
        for k = 1:c
            % Compute the mean of the 2x2 block
            sum_val = 0;
            for di = 0:1
                for dj = 0:1
                    sum_val = sum_val + double(center_512(2*i-1+di, 2*j-1+dj, k));
                end
            end
            result_256(i, j, k) = cast(sum_val / 4, class(img));
        end
    end
end

% Display the results
figure;
subplot(1,2,1); imshow(center_512); title('中心512x512区域');
subplot(1,2,2); imshow(result_256); title('下采样到256x256');
imwrite(result_256, 'Summer_camo_256.png');