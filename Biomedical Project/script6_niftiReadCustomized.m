function data = script6_niftiReadCustomized(filename)

    onState = warning('off', 'backtrace'); 
    c = onCleanup(@() warning(onState)); 

    data = niftiread(filename); % added lines: dicomread
    data = data(:,:,45);
   


end
