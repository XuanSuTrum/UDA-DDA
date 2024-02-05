clc;
clear;

%%session1 共计15名被试，每名被试共计15个trial，每个trail对应的de_LDS数据，将每个trial提取并合并在一起，其中de_LDS维度
%%为62*...*5,62通道5频带...个特征值，最终得到3394个特征值，同时注意，每个trial有一个标签，每个trial的label应该按照...维度
%%扩充，最终得到3394个label
data_folder = 'F:\Emotion_datasets\SEED_IV\eeg_feature_smooth\session1';   % 设置文件夹路径
data_files = dir(fullfile(data_folder, '*.mat'));  % 获取文件夹中所有的.mat文件
data_num_files = length(data_files);   % 获取文件数量

sort_nat_name=sort_nat({data_files.name});
data_cells = cell(data_num_files, 2); 

Label = load("F:\Emotion_datasets\SEED_IV\eeg_feature_smooth\session1_label.mat");
Label = Label.session1_label;

savepath = 'F:\Emotion_datasets\SEED_IV\feature_for_net_session1_LDS_de';
for i = 1:data_num_files
    data_file_path = fullfile(data_folder, data_files(i).name);  % 获取当前文件的完整路径
    all_data = load(data_file_path);  % 导入当前文件的数据
       
    matrices = {};
    label_   = {};
    % 循环遍历每个数据字段
    for j = 1:24
    % 生成字段名
       fieldName = sprintf('de_LDS%d', j);
    
    % 将数据存储到结构体的相应字段中
       matrices{j} = all_data.(fieldName);
       
       label_{j} = repmat(Label(j),size(all_data.(fieldName),2),1);
    end
    
    combinedMatrix = cat(2, matrices{:});
    label = cat(1,label_{:});
    combinedMatrix = permute(combinedMatrix,[2,1,3]);
    
    feature = reshape(combinedMatrix,852,310);
    
    dataset_session1.feature = feature;
    dataset_session1.label = label;
    
    fileName = [num2str(i) '.mat'];

    filePath = fullfile(savepath, fileName);

%     save(filePath, 'dataset_session1');

end