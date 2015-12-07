function [ Model ] = learnRecognition( training )

    trainingFeatures = [];
    trainingLabels = [];

    for i = 1 : size(training,2)
        imgset = training(i);
        for j = 1 : imgset.Count
            image = read(imgset, j);
            image = imresize(image, [120 120]);
            trainingFeatures = [trainingFeatures; extractHOGFeatures(image,'CellSize',[8 8])];
            
%             %
%             hogfeat=extractHOGFeatures(image,'CellSize',[8 8]);
%             image=rgb2gray(image);
%             points = detectSURFFeatures(image);
%             [features, validPoints] = extractFeatures(image,points);
%             features=features(:);
%             feat=[hogfeat features(1:200)'];
%             trainingFeatures=[trainingFeatures; feat];
            
            trainingLabels = [trainingLabels; i];
        end
    end
    
    %Model = fitcecoc(trainingFeatures, trainingLabels);
    Model = fitNaiveBayes(trainingFeatures, trainingLabels);
    %Model = fitcnb(trainingFeatures, trainingLabels);
    

end




