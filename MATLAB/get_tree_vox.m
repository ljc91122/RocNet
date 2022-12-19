function [vox,label,feas_all] = get_tree_vox(feas_all,label,vox_size)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

vox = zeros(vox_size,vox_size,vox_size);

% label = flip(label);
% feas_all = flip(feas_all,2);

if label(end)==0
    vox = 1;
    label = label(1:end-1);
    feas_all = feas_all(:,:,:,1:end-1);
    return;
elseif label(end)==1
    vox = 0;
    label = label(1:end-1);
    feas_all = feas_all(:,:,:,1:end-1);
    return;
elseif label(end)==2
    %vox = reshape(feas_all(:,:,:,end),size(vox)) >0.5;
    vox = reshape(feas_all(:,:,:,end),size(vox));
    label = label(1:end-1);
    feas_all = feas_all(:,:,:,1:end-1);
    return;
else
    
    label = label(1:end-1);
    
    [vox1,label,feas_all] = get_tree_vox(feas_all,label,vox_size/2);
    [vox2,label,feas_all] = get_tree_vox(feas_all,label,vox_size/2);
    [vox3,label,feas_all] = get_tree_vox(feas_all,label,vox_size/2);
    [vox4,label,feas_all] = get_tree_vox(feas_all,label,vox_size/2);
    [vox5,label,feas_all] = get_tree_vox(feas_all,label,vox_size/2);
    [vox6,label,feas_all] = get_tree_vox(feas_all,label,vox_size/2);
    [vox7,label,feas_all] = get_tree_vox(feas_all,label,vox_size/2);
    [vox8,label,feas_all] = get_tree_vox(feas_all,label,vox_size/2);
    
    vox(1:vox_size/2,1:vox_size/2,1:vox_size/2) = vox8;
    vox(1+vox_size/2:vox_size,1:vox_size/2,1:vox_size/2) = vox7;
    vox(1:vox_size/2,1+vox_size/2:vox_size,1:vox_size/2) = vox6;
    vox(1+vox_size/2:vox_size,1+vox_size/2:vox_size,1:vox_size/2) = vox5;
    vox(1:vox_size/2,1:vox_size/2,1+vox_size/2:vox_size) = vox4;
    vox(1+vox_size/2:vox_size,1:vox_size/2,1+vox_size/2:vox_size) = vox3;
    vox(1:vox_size/2,1+vox_size/2:vox_size,1+vox_size/2:vox_size) = vox2;
    vox(1+vox_size/2:vox_size,1+vox_size/2:vox_size,1+vox_size/2:vox_size) = vox1;
    
    
end

