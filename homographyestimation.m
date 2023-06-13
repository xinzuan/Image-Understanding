//  image size (pixel)
W1 = 1280; H1 = 960;
W2 = 640; H2 = 480;
//  Calibration matrix
K1 = [350 0 W1/2;
   0 350 H1/2;
   0 0 1]; //  pixel
K2 = [80 0 W2/2;
   0 80 H2/2;
   0 0 1]; //  pixel
//  Rotation matrix
thx = 10*pi/180; thy = -180*pi/180; thz = -70*pi/180; //  radian
Rx = [1 0 0; 0 cos(thx) -sin(thx); 0 sin(thx) cos(thx)];
Ry = [cos(thy) 0 sin(thy); 0 1 0; -sin(thy) 0 cos(thy)];
Rz = [cos(thz) -sin(thz) 0; sin(thz) cos(thz) 0; 0 0 1];
R1 = Rz*Ry*Rx;
thx = -10*pi/180; thy = -170*pi/180; thz = 0*pi/180; //  radian
Rx = [1 0 0; 0 cos(thx) -sin(thx); 0 sin(thx) cos(thx)];
Ry = [cos(thy) 0 sin(thy); 0 1 0; -sin(thy) 0 cos(thy)];
Rz = [cos(thz) -sin(thz) 0; sin(thz) cos(thz) 0; 0 0 1];
R2 = Rz*Ry*Rx;
//  Camera center in world coordinate frame (meter)
C1_ = [-2; -2; 10];
C2_ = [5; 5; 15];
//  Translation vector
t1 = -R1*C1_;
t2 = -R2*C2_;
//  Camera projection matrix (P = K*[R t])
P1 = K1*R1*[eye(3) -C1_];
P2 = K2*R2*[eye(3) -C2_];
//  3D points in world coordinate frame
x1 = [-10 -9 -7 -4 0 5 11]; y1 = [-10 -9 -7 -4 0 5 11]; z1 = 0;
[X1 Y1 Z1] = meshgrid(x1,y1,z1);
X1 = X1(:); Y1 = Y1(:); Z1 = Z1(:);
//  Draw 3D points and world coordinate frame
figure; plot3(X1,Y1,Z1,'k.'); hold on;
xlabel('X (meter)'); ylabel('Y (meter)'); zlabel('Z (meter)');
axis equal; grid on;
//  Draw camera coordinate frame
len = 2;
Xp = [len; 0; 0]; Yp = [0; len; 0]; Zp = [0; 0; len];
//  C2W rigid transform
CX1 = R1'*(Xp - t1); CY1 = R1'*(Yp - t1); CZ1 = R1'*(Zp - t1);
CX2 = R2'*(Xp - t2); CY2 = R2'*(Yp - t2); CZ2 = R2'*(Zp - t2);
plot3([C1_(1) CX1(1)],[C1_(2) CX1(2)],[C1_(3) CX1(3)],'b:','LineWidth',2);
plot3([C1_(1) CY1(1)],[C1_(2) CY1(2)],[C1_(3) CY1(3)],'g:','LineWidth',2);
plot3([C1_(1) CZ1(1)],[C1_(2) CZ1(2)],[C1_(3) CZ1(3)],'r:','LineWidth',2);
plot3([C2_(1) CX2(1)],[C2_(2) CX2(2)],[C2_(3) CX2(3)],'b-','LineWidth',2);
plot3([C2_(1) CY2(1)],[C2_(2) CY2(2)],[C2_(3) CY2(3)],'g-','LineWidth',2);
plot3([C2_(1) CZ2(1)],[C2_(2) CZ2(2)],[C2_(3) CZ2(3)],'r-','LineWidth',2);
hold off;
//  Acquire images of 3D points
x1 = P1*[X1'; Y1'; Z1'; ones(1,length(X1))];
x1(1,:) = x1(1,:)./x1(3,:); x1(2,:) = x1(2,:)./x1(3,:);
x1(3,:) = x1(3,:)./x1(3,:);
x2 = P2*[X1'; Y1'; Z1'; ones(1,length(X1))];
x2(1,:) = x2(1,:)./x2(3,:); x2(2,:) = x2(2,:)./x2(3,:);
x2(3,:) = x2(3,:)./x2(3,:);
//  Draw images of 3D points
figure;
subplot(121); plot(x1(1,:),x1(2,:),'r.'); axis equal; grid on; hold on;
axis([1 W1 1 H1]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
subplot(122); plot(x2(1,:),x2(2,:),'b.'); axis equal; grid on; hold on;
axis([1 W2 1 H2]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
save('data.mat','W1','H1','W2','H2','x1','x2');

// Add Gaussian noise to the 2D image points.
//  Load data
load('data.mat');
//  Add noise
scale = 7;
rng(1,'philox')
x1n = x1(1:2,:) + randn(size(x1(1:2,:)))*scale; x1n(3,:) = 1;
x2n = x2(1:2,:) + randn(size(x2(1:2,:)))*scale; x2n(3,:) = 1;

//  Draw data
figure;
subplot(121); plot(x1(1,:),x1(2,:),'r+'); axis equal; grid on; hold on;
plot(x1n(1,:),x1n(2,:),'b.'); axis([1 W1 1 H1]);
xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;
subplot(122); plot(x2(1,:),x2(2,:),'r+'); axis equal; grid on; hold on;
plot(x2n(1,:),x2n(2,:),'b.'); axis([1 W2 1 H2]);
xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;


// Estimate the homography using SVD, and check the result by transforming the 2D points.
//  Normalize data
iT1 = [W1+H1 0 W1/2; 0 W1+H1 H1/2; 0 0 1]; T1 = inv(iT1);
iT2 = [W2+H2 0 W2/2; 0 W2+H2 H2/2; 0 0 1]; T2 = inv(iT2);
x1nm = T1*x1n;
x2nm = T2*x2n;
//  create the matrix based on the point and its corresponding point
A = []; numPnt = size(x1nm,2);
for i=1:numPnt
   x = x1nm(1,i); y = x1nm(2,i); w = x1nm(3,i);
   x_ = x2nm(1,i); y_ = x2nm(2,i); w_ = x2nm(3,i);
   tmp = [0 0 0 -x*w_ -y*w_ -w*w_ x*y_ y*y_ w*y_;
   x*w_ y*w_ w*w_ 0 0 0 -x*x_ -y*x_ -w*x_];
   A = [A; tmp];
end
//  Estimate homography matrix using SVD
[U D V] = svd(A);
h = V(:,end);
Hnm = [h(1:3)'; h(4:6)'; h(7:9)'];
H = inv(T2)*Hnm*T1;
//  Transform points
x2n_ = H*x1n;
x2n_(1,:) = x2n_(1,:)./x2n_(3,:); x2n_(2,:) = x2n_(2,:)./x2n_(3,:);
x2n_(3,:) = x2n_(3,:)./x2n_(3,:);
x1n_ = inv(H)*x2n;
x1n_(1,:) = x1n_(1,:)./x1n_(3,:); x1n_(2,:) = x1n_(2,:)./x1n_(3,:);
x1n_(3,:) = x1n_(3,:)./x1n_(3,:);
//  Draw transformation result
figure;
subplot(121); plot(x1n(1,:),x1n(2,:),'b+'); axis equal; grid on; hold on;
plot(x2n(1,:),x2n(2,:),'r.'); axis equal; grid on; hold on;
axis([1 W1 1 H1]); xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;
subplot(122); plot(x2n_(1,:),x2n_(2,:),'b+'); axis equal; grid on; hold on;
plot(x2n(1,:),x2n(2,:),'r.'); axis equal; grid on; hold on;
axis([1 W2 1 H2]); xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;


// Add some outliers to the noise-contaminated 2D points

//  Outliers
rng(1,'philox')
outNum = 12;
xo = (randn(3,outNum)+1.5)*400;
yo = (randn(3,outNum)+1.5)*200;
xo(3,:) = 0.0010;
yo(3,:) = 0.0010;
x1out = [x1n xo];
y1out = [x2n yo];
//  Draw data
figure;
subplot(121); plot(x1(1,:),x1(2,:),'r+'); axis equal; grid on; hold on;
plot(x1n(1,:),x1n(2,:),'b.');
plot(xo(1,:),xo(2,:),'g.');axis([1 W1 1 H1]);
xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;
subplot(122); plot(x2(1,:),x2(2,:),'r+'); axis equal; grid on; hold on;
plot(x2n(1,:),x2n(2,:),'b.');
plot(yo(1,:),yo(2,:),'g.');axis([1 W2 1 H2]);
xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;


// Estimate the homography using RANSAC , and check the result by transforming the 2D points.                

function H = homography_svd(x1nm,x2nm,T1,T2)
   //  create the matrix based on the point and its corresponding point
   A = []; numPnt = size(x1nm,2);
   for i=1:numPnt
       x = x1nm(1,i); y = x1nm(2,i); w = x1nm(3,i);
       x_ = x2nm(1,i); y_ = x2nm(2,i); w_ = x2nm(3,i);
       tmp = [0 0 0 -x*w_ -y*w_ -w*w_ x*y_ y*y_ w*y_;
       x*w_ y*w_ w*w_ 0 0 0 -x*x_ -y*x_ -w*x_];
       A = [A; tmp];
   end
   //  Estimate homography matrix using SVD
   [U D V] = svd(A);
   h = V(:,end);
   Hnm = [h(1:3)'; h(4:6)'; h(7:9)'];
   H = inv(T2)*Hnm*T1;
end



 

//  Load data
load('data.mat');
//  Add noise
scale = 7;
rng(1,'philox')
x1n = x1(1:2,:) + randn(size(x1(1:2,:)))*scale; x1n(3,:) = 1;
x2n = x2(1:2,:) + randn(size(x2(1:2,:)))*scale; x2n(3,:) = 1;
//  Outliers
rng(1,'philox')
outNum = 20;
xo = (randn(3,outNum)+1.5)*400;
yo = (randn(3,outNum)+1.5)*200;
xo(3,:) = 0.0010;
yo(3,:) = 0.0010;
x1out = [x1n xo];
y1out = [x2n yo];
//  Draw data
figure;
subplot(121); plot(x1(1,:),x1(2,:),'r+'); axis equal; grid on; hold on;
plot(x1n(1,:),x1n(2,:),'b.');
plot(xo(1,:),xo(2,:),'g.');axis([1 W1 1 H1]);
xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;
subplot(122); plot(x2(1,:),x2(2,:),'r+'); axis equal; grid on; hold on;
plot(x2n(1,:),x2n(2,:),'b.');
plot(yo(1,:),yo(2,:),'g.');axis([1 W2 1 H2]);
xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;
//  Normalize data
//  x1nmo = x1out;
//  y2nmo = y1out;
x1nmo = T1*x1out;
y2nmo = T2*y1out;
N = 100;
s = 4;  //  Minimum number of points for estimating homography.
threshold = 10;
inlierRatio = .3;
[row, column] = size(x1nmo);
numInliersEachIteration = zeros(N,1);
storedModels = {};%zeros(N,3,3);
maxInliner =0;
maxInlierIndex =0;
Hransac=[];
for i = 1 : N
  
   %random subset of points
   subsetIndices = randsample(column, s);
   x_subset = T1* x1out(:,subsetIndices);
   y_subset = T2* y1out(:,subsetIndices);
      
   %fit a model to that subset
   Htemp = homography_svd(x_subset, y_subset,T1,T2);
  
    %transform the points using Htemp
   y1out_ = Htemp*x1out;
   y1out_(1,:) = y1out_(1,:)./y1out_(3,:);
   y1out_(2,:) = y1out_(2,:)./y1out_(3,:);
   y1out_(3,:) = y1out_(3,:)./y1out_(3,:);
  
   x1out_ = inv(Htemp)*y1out;
   x1out_(1,:) = x1out_(1,:)./x1out_(3,:);
   x1out_(2,:) = x1out_(2,:)./x1out_(3,:);
   x1out_(3,:) = x1out_(3,:)./x1out_(3,:);
  
   residuals = (y1out(1:2,:)-y1out_(1:2,:)).^2 + (x1out(1:2,:)-x1out_(1:2,:)).^2;
  
   [row, col, page]= find(residuals > threshold);     
   %record the number of inliers
   numInlier= length(inlierIndices);
   if numInlier > maxInliner
       maxInlierIndex = i;
       maxInliner = length(inlierIndices);
       x_inliers = T1*x1out(:,col);
       y_inliers = T2*y1out(:,col);
       Hransac  = homography_svd(x_inliers, y_inliers,T1,T2);
      
   end
end
y1out_ = Hransac*x1out;
y1out_(1,:) = y1out_(1,:)./y1out_(3,:); y1out_(2,:) = y1out_(2,:)./y1out_(3,:);
y1out_(3,:) = y1out_(3,:)./y1out_(3,:);
x1out_ = inv(Hransac)*y1out;
x1out_(1,:) = x1out_(1,:)./x1out_(3,:); x1out_(2,:) = x1out_(2,:)./x1out_(3,:);
x1out_(3,:) = x1out_(3,:)./x1out_(3,:);
//  Draw transformation result
figure;
subplot(121); plot(x1out(1,:),x1out(2,:),'b+'); axis equal; grid on; hold on;
plot(y1out(1,:),y1out(2,:),'r.');plot(xo(1,:),xo(2,:),'g.'); axis equal; grid on; hold on;
axis([1 W1 1 H1]); xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;
subplot(122); plot(y1out_(1,:),y1out_(2,:),'b+'); axis equal; grid on; hold on;
plot(y1out(1,:),y1out(2,:),'r.');plot(yo(1,:),yo(2,:),'g.'); axis equal; grid on; hold on;
axis([1 W2 1 H2]); xlabel('x (pixel)'); ylabel('y (pixel)'); hold off;







