% 保存变量为mat文件
% save("simout.mat", "simout")

% 22个过程变量pv + 19个成分变量cv

% 注意：任务1/2建议使用正常状态下的数据（数据的相关性分析）

clc, clear, close all;
% 加载数据
load("simout.mat");

%% 原始数据可视化 观察数据分布和数据质量问题（噪声 离群点等）
% 原始数据
index1 = 1; index2 = 2;
data_check1 = simout(:, index1);
data_check2 = simout(:, index2);

figure
% 折线图
subplot(3, 2, 1);
plot(data_check1); title(sprintf('变量%d 折线图', index1));
subplot(3, 2, 2);
plot(data_check2); title(sprintf('变量%d 折线图', index2));
% 直方图
subplot(3, 2, 3);
histogram(data_check1); title(sprintf('变量%d 直方图', index1));
subplot(3, 2, 4);
histogram(data_check2); title(sprintf('变量%d 直方图', index2));
% 箱体图
subplot(3, 2, 5);
boxplot(data_check1, 0, '.', 1); title(sprintf('变量%d 箱体图', index1));
subplot(3, 2, 6);
boxplot(data_check2, 0, '.', 1); title(sprintf('变量%d 箱体图', index2));
% % 填充区二维图
% subplot(2, 2, 4);
% area(data_check);

%% 小波去噪 + 平滑去噪
% 对所有数据去噪
% 小波去噪
simout_wde = wdenoise(simout);
% 平滑去噪
simout_dno = smoothdata(simout_wde, 'movmean', 60);  

% 不同去噪方法的效果对比展示
index = 23;
figure, plot(simout(:, index), '-g'), hold on;
plot(simout_wde(:, index), '-'), hold on;
plot(simout_dno(:, index), '.-r');
legend('原始数据', '小波去噪', '平滑去噪');
title('成分A数据去噪效果展示');


for index = 1:41
    figure
    plot(simout(:, index), '-'), hold on
    plot(simout_dno(:, index), '-*');
end

%% 综合去噪效果展示
% 23 21 18 16 3
flist = [3, 16, 18, 21, 23];
figure
for i = 1:numel(flist)
    subplot(5, 1, i); 
    plot(simout(:, flist(i))), hold on;
    plot(simout_dno(:, flist(i)), 'LineWidth', 3);
    legend('原始数据', '去噪后数据');
end

%% 归一化处理
% 最大最小归一化 [0, 1]
% simout_Nor = transpose(mapminmax(transpose(simout), 0, 1));
simout_Nor = mapminmax(simout_dno', 0, 1)';

% z分数规范化 针对矩阵的每一列
% 均值为0 方差为1
simout_dz = zscore(simout_dno);

% 不同规范化方法效果比较
index = 3;
figure;
subplot(3, 1, 1);
plot(simout_dno(:, index)); title('原数据');
subplot(3, 1, 2);
plot(simout_Nor(:, index)); title('最大最小归一化');
subplot(3, 1, 3);
plot(simout_dz(:, index)); title('z分数规范化');

%% 数据观测比较 可视化
% simout_dz_raw = zscore(simout);
figure
subplot(2, 1, 1);
boxplot(simout, 0, '.', 1);
xlabel('变量序号');
title('预处理前');
subplot(2, 1, 2);
boxplot(simout_dz, 0, '.', 1);
xlabel('变量序号');
title('预处理后');

%%
boxplot(simout, 0, '.', 1);
title('预处理前');

%% 变量预处理前后的比较
figure
num = 5;
for index = 1:num
    subplot(num, 1, index);
    plot(simout(:, index), '-'), hold on
    plot(simout_dno(:, index), '-*');
end

%% 主成分分析 PCA降维 对41个变量
% simout_dz = zscore(simout_dno);

R_22p19 = corrcoef(simout_dz);  
R_22p19 = abs(R_22p19);
% 热图
figure; heatmap(R_22p19, 'Fontsize', 10, 'Fontname', 'Yahei');
title('原41个变量的皮尔逊相关性矩阵');

% explained: 每个主成分解释的总方差的百分比
[coeff, score, latent, tsquared, explained, mu] = pca(simout_dz);
% 总方差百分比可视化
figure 
bar(explained);
title('41个主成分变量的总方差百分比');
xlabel('主成分序号');
ylabel('总方差百分比/贡献度');
set(gca, 'Ygrid', 'on');

cov_total = 0;
for i = 1:41
    cov_total = cov_total + explained(i);
    if cov_total >= 80
        break;
    end
end
% 主成分贡献度可视化
figure
bar(explained(1:i));
title('选取的主成分变量贡献度');
xlabel('主成分序号');
ylabel('贡献度');
set(gca, 'Ygrid', 'on');

% 经pca变换后的41个主成分
data_pc = (coeff * simout_Nor')';

% 热图 皮尔逊相关系数
R_41 = corrcoef(data_pc);
R_41 = abs(R_41);
figure; heatmap(R_41, 'Fontsize', 10, 'Fontname', 'Yahei');
title('经pca处理后41个变量的相关系数矩阵');

%% 22个过程变量的主成分分析
% 仅需做零均值化
% simout_dz = detrend(simout_dno);
% simout_dz = zscore(simout_dno);

% explained: 每个主成分解释的总方差的百分比
[coeff, score, latent, tsquared, explained, mu] = pca(simout_dz(:, 1:22));
% 总方差百分比可视化
bar(explained);
title('22个主成分的总方差百分比');
xlabel('主成分序号');
ylabel('总方差百分比/贡献度');
set(gca, 'Ygrid', 'on');    

% 经pca变换后的22个主成分（过程变量）
data_pca_pv22 = (coeff * simout_Nor(:, 1:22)')';

% 热图
R_22 = corrcoef(data_pca_pv22);
R_22 = abs(R_22);
figure; heatmap(R_22, 'Fontsize', 10, 'Fontname', 'Yahei');
title('经pca处理后22个过程变量的相关系数矩阵');

%% 从22个过程变量中筛选具有代表性的主成分
% 查找最后一个非零元素的索引值 即非零元素的数量
num = find(explained>3, 1, 'last');
% 获得主成分的总方差百分比
explained_pc = explained(1:num, 1);
data_pv_num = data_pca_pv22(:, 1:num);

% 主要成分的总方差百分比可视化
figure
bar(explained_pc);
title('主要成分的总方差百分比');
xlabel('主成分序号');
ylabel('总方差百分比/贡献度');
set(gca, 'Ygrid', 'on');

% 热图
R_num = corrcoef(data_pv_num);
R_num = abs(R_num);
figure; heatmap(R_num, 'Fontsize', 10, 'Fontname', 'Yahei');
title('num个主成分过程变量的相关系数矩阵');

%% 任务2 参数预测
%% 22个过程变量的相关性分析
% 热图
R_pv22 = corrcoef(simout_dz(:, 1:22));
R_pv22 = abs(R_pv22);
figure; heatmap(R_pv22, 'Fontsize', 10, 'Fontname', 'Yahei');
title('原22个过程变量的相关系数矩阵');

% 一些过程变量之间存在高度的相关性

%% 变量相关性分析 协方差 相关系数
%% 成分变量与原过程变量之间的相关性分析
R_22p19 = corrcoef(simout_dz);
R_22p19 = abs(R_22p19);
% 热图
figure; heatmap(R_22p19, 'Fontsize', 10, 'Fontname', 'Yahei');
title('成分变量与原过程变量之间的相关性矩阵');

% 保留下三角，上三角清零（包括对角线）
R_22p19 = R_22p19.*tril(ones(41, 41), -1);
eta = 0.8; % 阈值
[row, col] = find(R_22p19 > eta); % 相关度高于阈值的变量组
val_group = [row, col];

% 强关联的变量组
% 2, 3
% 5, 6, 11, 18, 20, 22
% 7, 13, 16

%% 成分变量与原过程变量之间的协方差矩阵
cov_22p19 = cov(simout_dz);
cov_22p19 = abs(cov_22p19);
figure; heatmap(cov_22p19, 'Fontsize', 10, 'Fontname', 'Yahei');
title('成分变量与原过程变量之间的协方差矩阵');

%% 成分变量与经pca处理后过程变量之间的相关性分析
data_pca22p19 = [data_pca_pv22, simout_dz(:, 23:41)];
R_pca22p19 = corrcoef(data_pca22p19);
R_pca22p19 = abs(R_pca22p19);
% 热图
figure; heatmap(R_pca22p19, 'Fontsize', 10, 'Fontname', 'Yahei');
title('成分变量与经pca处理后过程变量之间的相关性矩阵');

% 保留下三角，上三角清零（包括对角线）
R_pca22p19 = R_pca22p19.*tril(ones(41, 41), -1);
eta = 0.8; % 阈值
[row, col] = find(R_pca22p19 > eta); % 相关度高于阈值的变量组
val_group = [row, col];

%% 
% 针对成分变量A(index=23) 确定过程测量变量
r_cv23 = R_pca22p19(23, 1:22);
[r_cv23, sort_idx] = sort(r_cv23, 'descend');
% 22个pca过程变量的相关性降序排序
figure, bar(r_cv23);
labels = {};
for i = 1:22
%     labels(i) = 'pv' + num2str(sort_idx(i));
    labels(i) = {sprintf('pv%d', sort_idx(i))};
end
xticks(1:22);
xticklabels(labels);
title('22个pca过程变量的相关性降序排序');

r_cv23_prime = R_22p19(23, 1:22);
[r_cv23_prime, sort_idx] = sort(r_cv23_prime, 'descend');
% 22个原过程变量的相关性降序排序
figure, bar(r_cv23_prime);
for i = 1:22
%     labels(i) = 'pv' + num2str(sort_idx(i));
    labels(i) = {sprintf('pv%d', sort_idx(i))};
end
xticks(1:22);
xticklabels(labels);
xlabel('过程变量序号');
ylabel('相关性');
title('22个原过程变量的相关性降序排序');

% 选取前nval个相关性系数最大的变量作为过程测量变量
nval = 6;
main_measure = data_pca_pv22(:, sort_idx(1:nval));
% figure, plot(main_measure);

%% 回归分析
%% 多元回归
X = [ones(7201, 1), main_measure];
y = simout_dz(:, 23);

% 最小二乘回归分析
model_ls = fitlm(X, y);
yfit_ls = model_ls.Fitted;
% 岭回归分析
model_ridge = fitrlinear(X, y, 'Regularization', 'ridge');
yfit_ridge = predict(model_ridge, X);
% lasso回归分析
model_ridge = fitrlinear(X, y, 'Regularization', 'lasso');
yfit_lasso = predict(model_ridge, X);

model_net = feedforwardnet([10], 'traingdm');
% hiddenLayerSize = 10; % 隐含层大小
% net = patternnet(hiddenLayerSize); % 创建网络
model_net.trainFcn = 'trainscg';  % 使用 Scaled conjugate gradient backpropagation
model_net.trainParam.lr = 0.01;
model_net.trainParam.epochs = 500;
model_net.trainParam.goal = 1e-3;      % 设置目标误差
model_net.trainParam.max_fail = 6;
model_net = train(model_net, X', y');
yfit_net = model_net(X')';

figure
subplot(4, 1, 1);
plot(1:7201, y, '-', 1:7201, yfit_ls, '-.r');
title('成分变量A：最小二乘回归模型');
xlabel('样本');
ylabel('成分变量A');
legend('实际值', '预测值');

subplot(4, 1, 2);
plot(1:7201, y, '-', 1:7201, yfit_ridge, '-.r');
title('成分变量A：岭回归模型');
xlabel('样本');
ylabel('成分变量A');
legend('实际值', '预测值');

subplot(4, 1, 3);
plot(1:7201, y, '-', 1:7201, yfit_lasso, '-.r');
title('成分变量A：lasso回归模型');
xlabel('样本');
ylabel('成分变量A');
legend('实际值', '预测值');

subplot(4, 1, 4);
plot(1:7201, y, '-', 1:7201, yfit_net, '-.r');
title('成分变量A：BP回归模型');
xlabel('样本');
ylabel('成分变量A');
legend('实际值', '预测值');

% 评价指标
% R方
R_sq_ls = 1 - sum((y - yfit_ls).^2) / sum((y - mean(y)).^2)
R_sq_ridge = 1 - sum((y - yfit_ridge).^2) / sum((y - mean(y)).^2)
R_sq_lasso = 1 - sum((y - yfit_lasso).^2) / sum((y - mean(y)).^2)
R_sq_net = 1 - sum((y - yfit_net).^2) / sum((y - mean(y)).^2)
% MSE
mse_ls = sum((y - yfit_ls).^2) / length(y)
mse_ridge = sum((y - yfit_ridge).^2) / length(y)
mse_lasso = sum((y - yfit_lasso).^2) / length(y)
mse_net = sum((y - yfit_net).^2) / length(y)





