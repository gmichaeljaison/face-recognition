classdef (Abstract) FaceRecognition
    % Interface for FaceRecognition system 
    
    properties
        Model
    end
    
    methods(Abstract, Static)
        Model = train
        x = predict
    end
    
end
