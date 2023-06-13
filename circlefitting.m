%  Generating points on a circle
th = linspace(0,2*pi,30)';
r = 2.5;      %  radius of circle
ox = 3.3;     %  center of circle in x-axis
oy = 7.7;     %  center of circle in y-axis
scale = 0.1; %  noise level
x = r*cos(th) + ox + randn(size(th))*scale;
y = r*sin(th) + oy + randn(size(th))*scale;
figure; plot(x, y, 'r.'); axis equal;

%  Estimate circle parameters using pseudoinverse.
X = [x y ones(length(x),1)];
Y = [(x.^2+y.^2)];
p = inv(X'*X)*X'*Y;
x2 = p(1)/2;
y2 = p(2)/2;
r2 = sqrt(4*p(3)+p(1)^2+p(2)^2)/2;
figure;plot(x, y, 'r.');hold on;axis equal;
plot(r2*cos(th)+x2,r2*sin(th)+y2,'g','linewidth',1); hold off;

%hold off;

%  Estimate circle parameters using SVD.

X = [x y x.^2+y.^2 ones(length(x),1)];

[U D V] = svd(X)
p3 = V(:,end);
p3= -(p3/p3(3));
x3 = p3(1)/2;
y3 = p3(2)/2;
r3 = sqrt(4*p3(4)+p3(1)^2+p3(2)^2)/2;
figure;plot(x, y, 'r.');hold on;
plot(r3*cos(th)+x3,r3*sin(th)+y3,'c','linewidth',1); hold off;
axis equal;

%  Add some outliers and estimate circle parameters using RANSAC.

%  Outliers
outNum = 20;
xo = (rand(outNum,1)+0.5)*12;
yo = (rand(outNum,1)+0.5)*16;
xo = [x; xo];
yo = [y; yo];

p = 0.99;
s = 3; e = outNum/length(xo);
N = log(1-p) / log(1-(1-e)^s);
thresh = 2*outNum*mean(yo);
max = 0;


for i=1:ceil(N)
    id = randperm(length(xo),s);
    xi = xo(id);
    yi = yo(id);
    X4 = [xi yi xi.^2+yi.^2 ones(s,1)];

    [U1 D1 V1] = svd(X4);
    p4 = V1(:,end);
    p4= -(p4/p4(3));
    x4 = p4(1)/2;
    y4 = p4(2)/2;
    xi = x4*xo;
    yi = y4*yo

    err = abs(sqrt((ox-xi).^2+(oy-yi).^2)-yo);
    cnt = sum(err<thresh);
    if cnt > max
        max = cnt;
        pmax = p4;
    end
end
err2 = abs(sqrt((ox-pmax(1)*xo).^2+(oy-pmax(2)*yo).^2)-yo);

id = find(err2<thresh);
X5 = [xo(id) yo(id) xo(id).^2+yo(id).^2 ones(length(id),1)];
[U2 D2 V2] = svd(X5)
pRANSAC = V2(:,end);
pRANSAC= -(pRANSAC/pRANSAC(3));


x4 = pRANSAC (1)/2;
y4 = pRANSAC (2)/2;
r4 = sqrt(4*pRANSAC (4)+pRANSAC (1)^2+pRANSAC (2)^2)/2;

figure; plot(r4*cos(th)+x4,r4*sin(th)+y4,'g','linewidth',1); hold on;
plot(xo, yo, 'bo'); hold off;

axis equal;

%(You need at least three points to estimate circle parameters)



