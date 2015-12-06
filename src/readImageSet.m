function [images, Y] = readImageSet(imageset)

    Y = {};
    images = {};

    n = 1;
    for i = 1 : size(imageset,2)
        imgset = imageset(i);
        for j = 1 : imgset.Count
            image = read(imgset, j);
            images{n} = image;
            Y{n} = imgset.Description;
            n = n + 1;
        end
    end
    
    
%     Model = fitcecoc(trainingFeatures, trainingLabels);

end
