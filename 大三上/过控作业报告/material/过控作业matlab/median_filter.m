% function filtered_image = median_filter(input_image, filter_size)
% [m, n] = size(Image);
% G = zeros(m,n);
% for x=1:m
%     for y = 1:n
%         if (x==1||y==1||x==m||y==n)
%             G(x,y)=Image(x,y);
%         else  % 选出第5大的数
%             H = sort([Image(x-1,y-1), Image(x-1,y),Image(x-1,y+1),Image(x,y),... 
%                 Image(x,y+1),Image(x+1,y-1),Image(x+1,y),Image(x+1,y+1)]);
%             G(x,y)=median(H);
%         end
%     end
% end






function filtered_image = median_filter(input_image, filter_size)
    % 输入参数：
    % input_image - 输入图像（灰度图）
    % filter_size - 滤波器大小，必须是奇数

    % 输出参数：
    % filtered_image - 经过中值滤波处理后的图像

    % 获取输入图像的尺寸
    [rows, cols] = size(input_image);

    % 初始化输出图像
    filtered_image = zeros(rows, cols);

    % 计算滤波器半径
    radius = floor(filter_size / 2);

    % 对每个像素进行中值滤波
    for i = radius + 1:rows - radius
        for j = radius + 1:cols - radius
            % 提取当前像素周围的邻域
            neighborhood = input_image(i - radius:i + radius, j - radius:j + radius);

            % 对邻域进行排序并取中值
            sorted_neighborhood = sort(neighborhood(:));
            median_value = sorted_neighborhood((filter_size * filter_size + 1) / 2);

            % 将中值赋给输出图像的对应像素
            filtered_image(i, j) = median_value;
        end
    end
    
%     % 对图像边缘进行额外处理
%     for x=1:rows
%         for y = 1:cols
%             if (x <= radius || y <= radius || x >= rows-radius || y >= cols-radius)
%                     filtered_image(x,y)=input_image(x,y);
%             end
%         end
%     end
end
