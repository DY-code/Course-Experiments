clc, clear

% 加载数据
simout_label1 = load("simout.mat").simout;
simout_label2 = load("simout_label1.mat").simout;
simout_label3 = load("simout_label8.mat").simout;
simout_label4 = load("simout_label13.mat").simout;
simout_label5 = load("simout_label14.mat").simout;
simout_label6 = load("simout_label19.mat").simout;
simout_label7 = load("simout_label5.mat").simout;
simout_label8 = load("simout_label4.mat").simout;
simout_label9 = load("simout_label3.mat").simout;
simout_label10 = load("simout_label2.mat").simout;

%% 小波去噪 + 平滑去噪
% 对所有数据去噪
% 小波去噪
simout_label1_dno = wdenoise(simout_label1);
simout_label2_dno = wdenoise(simout_label2);
simout_label3_dno = wdenoise(simout_label3);
simout_label4_dno = wdenoise(simout_label4);
simout_label5_dno = wdenoise(simout_label5);
simout_label6_dno = wdenoise(simout_label6);
simout_label7_dno = wdenoise(simout_label7);
simout_label8_dno = wdenoise(simout_label8);
simout_label9_dno = wdenoise(simout_label9);
simout_label10_dno = wdenoise(simout_label10);
% 平滑去噪
simout_label1_dno = smoothdata(simout_label1_dno, 'movmean', 60);  
simout_label2_dno = smoothdata(simout_label2_dno, 'movmean', 60);
simout_label3_dno = smoothdata(simout_label3_dno, 'movmean', 60);
simout_label4_dno = smoothdata(simout_label4_dno, 'movmean', 60);
simout_label5_dno = smoothdata(simout_label5_dno, 'movmean', 60);
simout_label6_dno = smoothdata(simout_label6_dno, 'movmean', 60);
simout_label7_dno = smoothdata(simout_label7_dno, 'movmean', 60);
simout_label8_dno = smoothdata(simout_label8_dno, 'movmean', 60);
simout_label9_dno = smoothdata(simout_label9_dno, 'movmean', 60);
simout_label10_dno = smoothdata(simout_label10_dno, 'movmean', 60);

%%
% index = 23;
% figure, plot(simout_label1(:, 23), '-'), hold on;
% plot(simout_label1_dno(:, 23), '.-');
% legend('原始数据', '去噪后数据');
% title('成分A数据去噪效果展示');

% for index = 1:41
%     figure
%     plot(simout_label1(:, index), '-'), hold on
%     plot(simout_label1_dno(:, index), '-*');
% end

% for index = 1:41
%     figure
%     plot(simout_label2(:, index), '-'), hold on
%     plot(simout_label2_dno(:, index), '-*');
% end

% for index = 1:41
%     figure
%     plot(simout_label4(:, index), '-'), hold on
%     plot(simout_label4_dno(:, index), '-*');
% end

%% 归一化处理
% z分数规范化 针对矩阵的每一列
% 均值为0 方差为1
simout_label1_dz = zscore(simout_label1_dno);
simout_label2_dz = zscore(simout_label2_dno);
simout_label3_dz = zscore(simout_label3_dno);
simout_label4_dz = zscore(simout_label4_dno);
simout_label5_dz = zscore(simout_label5_dno);
simout_label6_dz = zscore(simout_label6_dno);
simout_label7_dz = zscore(simout_label7_dno);
simout_label8_dz = zscore(simout_label8_dno);
simout_label9_dz = zscore(simout_label9_dno);
simout_label10_dz = zscore(simout_label10_dno);

% 使用未去噪的数据进行归一化
% simout_label1_dz = zscore(simout_label1);
% simout_label2_dz = zscore(simout_label2);
% simout_label3_dz = zscore(simout_label3);
% simout_label4_dz = zscore(simout_label4);
% simout_label5_dz = zscore(simout_label5);

%%
% index = 3;
% figure;
% subplot(2, 1, 1);
% plot(simout_label1_dno(:, index)); title('原数据');
% subplot(2, 1, 2);
% plot(simout_label1_dz(:, index)); title('z分数规范化');

%% 划分训练集 测试集
data = [simout_label1_dz, ones(7201, 1);
            simout_label2_dz, ones(7201, 1)*2;
            simout_label3_dz, ones(7201, 1)*3;
            simout_label4_dz, ones(7201, 1)*4;
            simout_label5_dz, ones(7201, 1)*5;
            simout_label6_dz, ones(7201, 1)*6;
            simout_label7_dz, ones(7201, 1)*7;
            simout_label8_dz, ones(7201, 1)*8;
            simout_label9_dz, ones(7201, 1)*9;
            simout_label10_dz, ones(7201, 1)*10;];
% 数据集总量
n = size(data, 1);
% 打乱顺序
idxr = randperm(n);
% idxr = 1:n; % 不打乱顺序
% 训练集/测试集划分
idx_div = floor(n*0.8);
train_set = data(idxr(1:idx_div), :);
test_set = data(idxr(idx_div+1:n), :);
% 预测数据/标签
train_X = train_set(:, 1:41);
train_y = train_set(:, 42);
test_X = test_set(:, 1:41);
test_y = test_set(:, 42);

% 标签转换为独热编码
train_y_onehot = full(ind2vec(train_y'));
% test_y_onehot = full(ind2vec(test_y'));

%% BP神经网络
% net = feedforwardnet([10], 'traingdm');
hiddenLayerSize = [35, 20]; % 隐含层大小
net = patternnet(hiddenLayerSize); % 创建网络
net.trainFcn = 'trainscg';  % 使用 Scaled conjugate gradient backpropagation
net.trainParam.lr = 0.01;
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-3;      % 设置目标误差
net.trainParam.max_fail = 6;
% net = train(net, train_X', train_y');
[net, tr] = train(net, train_X', train_y_onehot);

%% 保存和加载模型文件
% save('net.mat', "net");
% load('net.mat');
% sim(net, test_X')

%%
% preds = net(test_X');
% preds_train = net(train_X');

y_pred_onehot = net(test_X');
y_train_pred_onehot = net(train_X');
% 独热编码转换为整数标签
preds = vec2ind(y_pred_onehot);
preds_train = vec2ind(y_train_pred_onehot);

% 模型输出四舍五入得到预测标签
% preds = round(preds);
% preds(preds<1) = 1; preds(preds > 5) = 5;
% preds_train = round(preds_train);
% preds_train(preds_train<1) = 1; preds_train(preds_train > 5) = 5;

% figure
% plot(preds); hold on
% plot(test_y);
% legend('preds', 'test\_y');

% 在训练集和测试集上的准确率
prec_train = sum(preds_train' == train_y) / size(train_y, 1)
prec = sum(preds' == test_y) / size(test_y, 1)

%  混淆矩阵
figure
cm = confusionchart(preds_train, train_y);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(preds, test_y);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';







