%%
global camlist
global cam
global faceDetector
camlist = webcamlist;
cam = webcam(camlist{1});

global labelname
global Model;
global numperson;

%%
 imgdb = imageSet('../data/owndata', 'recursive');
Model = learnRecognition(imgdb);

%%
 faceDetector = vision.CascadeObjectDetector;
shapeInserter = vision.ShapeInserter('BorderColor','Custom', ...
    'CustomBorderColor',uint8([255 255 0]));

%%  Key interrupt
global keypress
keypress=0;
gcf
set(gcf, 'KeyPressFcn', @captureface)

%%
numperson=3;
labelname=cellstr('shreyas');
labelname{2}=('Michael');
labelname{3}=('Dev');
%figure; hold on;
%%
while (true)
    imshot = snapshot(cam);
   % imshot = imresize(imshot, 0.3);
%    imshot1=imshot;
%    threshold=1;
%     imshot1(find(abs(imshot1-background) <= threshold) ) = 0;
    bboxes = step(faceDetector, imshot);
    if (numel(bboxes) == 0)
        imshow(imshot);
        continue;
    end
    
    for i = 1 : size(bboxes,1)
        bbox = bboxes(i,:);
        faceimg = imcrop(imshot, bbox);
    
        faceimg = imresize(faceimg, [120 120]);
        faceFeature = extractHOGFeatures(faceimg,'CellSize',[8 8]);
        
%         image = imresize(faceimg, [200 200]);
%          hogfeat=extractHOGFeatures(image,'CellSize',[8 8]);
%             image=rgb2gray(image);
%             points = detectSURFFeatures(image);
%             [features, validPoints] = extractFeatures(image,points);
%             features=features(:);
%             faceFeature=[hogfeat features(1:200)'];
%             if(size(faceFeature,2) == Model.NDims)
        
        %[label,score,cost] = predict(obj,X)
        %[label,score,cost] = predict(Model, faceFeature);
        [label] = predict(Model, faceFeature);
        %[label,Posterior,cost] = predict(Model, faceFeature);
%             end
        % disp([label,score]);
       disp(label);
       %disp(score);
%         if(label==1)
%             label='Shreyas';
%         end
%          if(label==2)
%             label='Michael';
%          end
%          if(label==3)
%             label='Dev';
%         end
        bboxTmp = reshape(bbox, [2 2])';
        imshot = insertText(imshot, bboxTmp(1,:), labelname{label});
    
%         rectangle('Position', bbox, 'LineWidth',2, 'EdgeColor','y');
        imshot = step(shapeInserter, imshot, int16(bbox));
    end
    imshow(imshot);
    
    pause(0.01);
end