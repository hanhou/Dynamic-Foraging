%% Some quick and dirty code here
%% Motif optimization
p_bait = @(p,n) 1-(1-p).^n;
m = 1:50;
n = 1:50;
[mm, nn] = meshgrid(m, n);
ratios = 2.^(0:5);

figure(3); clf
subplot(1,2,1); hold on;
set(gcf,'uni','norm','pos',[0.15       0.313         0.5       0.357]);

h = plot(0,0,'bo','markersize',10,'markerfacecolor','b');
plot([0 1],[0 1],'k--')
plot([0,1,1,0],[0,0,1,1],'k')
axis square

for ratio = ratios
    plot([0,1/ratio],[0,1],'b:')
    text(1/ratio,1,sprintf('%g:1',ratio),'color','b','rotation',rad2deg(atan(ratio)))
end
xlabel('p_0')
ylabel('p_1')

% daObj=VideoWriter('optimal_p','MPEG-4'); %my preferred format
% open(daObj);

for ratio = ratios
    for p1 = 0.01:0.05:0.9
        p0 = p1/ratio;
%         
%         p1 = 0.01;
%         p0 = 0.0011;
                  
        set(h,'xdata',p0,'ydata',p1);
        
        p_motif = ((mm-1)*p1 + p_bait(p0, mm+1) + (nn-1)*p0 + p_bait(p1, nn+1))./(mm+nn);

        max_p = max(p_motif(:));
        opti_m = mm(p_motif == max_p);
        opti_n = nn(p_motif == max_p);
        
        subplot(1,2,2); cla
        surfc(mm, nn, p_motif); 
        % clabel(c,h,'fontsize',15)
        hold on;
        plot3(opti_m, opti_n, max_p, 'pb','markersize',30,'markerfacecolor','b')

        colormap('hot')
        colorbar
        xlabel('m')
        ylabel('n')
        SetFigure(15)
        title(sprintf('   < r | \\{m,n\\} >_{(%4.3f, %4.3f)}     Optimal \\{m^*,n^*\\} = \\{%g, %g\\}', p1,p0,opti_m, opti_n))
        
        plot3([opti_m, opti_m], [opti_n,opti_n],[min(zlim(gca)) max_p],'b--','linew',1.5)
        plot3([opti_m, opti_m], [opti_n,opti_n],[min(zlim(gca)) min(zlim(gca))],'bo','markersize',10, 'markerfacecolor','b')
        
        % ideal-pHat-Greedy
        m_star_pHatgreedy = floor(log(1-p1)/log(1-p0));
        p_star_pHatgreedy = ((m_star_pHatgreedy-1)*p1 + p_bait(p0, m_star_pHatgreedy+1) + p_bait(p1, 2))/(m_star_pHatgreedy + 1);
%         plot3(m_star_pHatgreedy, 1, p_star_pHatgreedy, 'pr','markersize',15,'markerfacecolor','r')
%         plot3([m_star_pHatgreedy, m_star_pHatgreedy], [1,1],[min(zlim(gca)) p_star_pHatgreedy],'r--','linew',1.5)
%         plot3([m_star_pHatgreedy], [1],[min(zlim(gca))],'ro','markersize',7, 'markerfacecolor','r')
        
%         temp = (1-p1^2-(1-p0).^m)./m;
%         m_optim_simple = m_star(p0,p1);
%         plot3(m_optim_simple, 1, min(zlim(gca)),'bo','markersize',15)
        
        view(-30,10)
        drawnow;
        
        writeVideo(daObj,getframe(gcf)); %use figure, since axis changes size based on view
    end
end

% clean up
% close(daObj);


% Set up recording parameters (optional), and record
% OptionZ.FrameRate=15;OptionZ.Duration=5.5;OptionZ.Periodic=true;
% CaptureFigVid([-5,10;-85,10;-5,10], 'WellMadeVid',OptionZ)


%% m*
p0 = 0.001:0.01:1;
p1 = 0.001:0.01:1;
[pp0, pp1] = meshgrid(p0, p1);
pp0 = tril(pp0,1);
pp1 = tril(pp1,1);

[m_star, n_star, p_star] = m_star_optimal(pp0,pp1); % Ideal-pHat-Optimal

n_star_larger_than_one = find(n_star>1);

fprintf('Total n_star>1: %g\n',length(n_star_larger_than_one));
for i = 1:length(n_star_larger_than_one)
    fprintf('Found n_star>1: p = (%g,%g), (m,n) = (%g,%g)\n', pp0(n_star_larger_than_one(i)), pp1(n_star_larger_than_one(i)),...
                                                            m_star(n_star_larger_than_one(i)), n_star(n_star_larger_than_one(i)));
end

%% Test Lagrange
% p0 = 0.1;
% p1 = 0.7;
% m_star_greedy = floor(log(1-p1)./log(1-p0))
% [m_star_, n_star, ~] = m_star_optimal(p0,p1)
% mm = 1:50;
% nn = 50-mm;
% plot(mm, p1-log(1-p0)*(1-p0).^(mm+1),'o-'); hold on
% plot(mm, p0-log(1-p1)*(1-p1).^(nn+1),'o-');


%%
m_star_greedy = floor(log(1-pp1)./log(1-pp0)); % Ideal-pHat-greedy
p_star_greedy = (2+(m_star_greedy-1).*pp1-(1-pp1).^(1+1)-(1-pp0).^(m_star_greedy+1))./(m_star_greedy+1);
% (1-(1-pp0).^(m_star_greedy+1)-pp1.^2)./(m_star_greedy+1) + pp1;
%((m_star_greedy-1).*pp1 + 1-(1-pp0).^(m_star_greedy+1) + 1-(1-pp1).^2)./(m_star_greedy + 1);
  %(1-(1-pp0).^(m_star_greedy+1)-pp1.^2)./(m_star_greedy+1) + pp1;



figure(1); clf; set(gcf,'uni','norm','pos',[0.118       0.297       0.748       0.433]);
subplot(1,3,1)
[c,h] = contourf (pp0, pp1, m_star,[0, 1, 2, 3, 2.^(2:6), 100], 'linecolor','k');
clabel(c,h,'fontsize',15)
        
colormap('cool')
colorbar
xlabel('p_0')
ylabel('p_1')
axis square
SetFigure(15)
plot_bari()
set(findall(gcf,'type','axes'),'ytick',get(gca,'xtick'))
title('m*_{IdealpHatOptimal}')

subplot(1,3,2)
pcolor_with_contour(pp0, pp1, abs(m_star-m_star_greedy), gca, [0:5:20,100]); caxis([0,20])
% [c,h] = contourf(pp0, pp1, m_star-m_approx,[0:20,100], 'linecolor','k');
% clabel(c,h,'fontsize',15)
        
colormap(gca,flipud(winter))
colorbar
xlabel('p_0')
ylabel('p_1')
axis square
SetFigure(15)
plot_bari()
set(findall(gcf,'type','axes'),'ytick',get(gca,'xtick'))
title('m*_{IdealpHatOptimal} - m*_{IdealpHatGreedy}')

subplot(1,3,3)
pcolor_with_contour(pp0, pp1, p_star_greedy./p_star, gca,[min(min(p_star_greedy./p_star)),1]); caxis([min(min(p_star_greedy./p_star)),1])
% [c,h] = contourf(pp0, pp1, m_star-m_approx,[0:20,100], 'linecolor','k');
% clabel(c,h,'fontsize',15)
        
colormap(gca,'winter')
colorbar
xlabel('p_0')
ylabel('p_1')
axis square
SetFigure(15)
plot_bari()
set(findall(gcf,'type','axes'),'ytick',get(gca,'xtick'))
title('<p*_{IdealpHatGreedy}>/<p*_{IdealpHatOptimal}> ')



%% Relative Benefits from baiting (the ugly term)

relative_baiting_benefit_optimal = (p_star - pp1)./pp1;
relative_baiting_benefit_greedy = (p_star_greedy - pp1)./pp1; %(1-(1-pp0).^(m_star+1)-pp1.^2)./(m_star+1)./pp1;
relative_baiting_benefit_2ndOrder = (pp0 - pp0.*pp1/2)./pp1;

% relative_baiting_benefit = (1-(1-pp0).^ceil(pp1./pp0)-pp1.^2)./ceil(pp1./pp0)./pp1;

figure(2); clf;
set(gcf,'uni','norm','pos',[0.118       0.297       0.748       0.433]);

subplot(1,3,1)
pcolor_with_contour(tril(pp0), tril(pp1), tril(relative_baiting_benefit_optimal),gca, [0,0.01 * 2.^(0:10)]);

colormap('cool')
xlabel('p_0')
ylabel('p_1')
axis square
title('Relative benefit from baiting (pHatOptimal)')
plot_bari()


subplot(1,3,2)
pcolor_with_contour(tril(pp0), tril(pp1), tril(relative_baiting_benefit_greedy),gca, [0,0.01 * 2.^(0:10)]);

colormap('cool')
xlabel('p_0')
ylabel('p_1')
axis square
title('Relative benefit from baiting (pHatGreedy, the 2nd term / p_1)')
plot_bari()

subplot(1,3,3)
pcolor_with_contour(tril(pp0), tril(pp1), tril(relative_baiting_benefit_2ndOrder),gca,[0,0.01 * 2.^(0:10)]);
title('2nd order approximation')
plot_bari()

axis square
SetFigure(15)
set(findall(gcf,'type','axes'),'ytick',get(gca,'xtick'))


%% Matching slope and prob_matching_index
%{
% m = floor(log(1-pp1)./log(1-pp0));
[m_star, n_star, p_star]  = m_star_optimal(pp0,pp1);


log_c_ratio = log(m_star);
log_r_ratio = log(((m_star+1).*pp1 - pp1.^2)./(1-(1-pp0).^(m_star+1)));

frac_c_ratio = m_star./(m_star+1); 
frac_r_ratio = ((m_star+1).*pp1 - pp1.^2)./(1-(1-pp0).^(m_star+1)+(m_star+1).*pp1 - pp1.^2);

% matching_slope = log_c_ratio./log_r_ratio;
matching_slope = frac_c_ratio./frac_r_ratio;
% matching_slope(log_c_ratio == 0) = 1;

figure(3); clf;
set(gcf,'uni','norm','pos',[0.303       0.283       0.519       0.434]);

subplot(1,2,1)
pcolor_with_contour(tril(pp0), tril(pp1), tril(matching_slope),gca,[]);
title('Matching slope (frac C /frac R)')
axis square; colorbar
colormap('jet'); caxis([0,2])
xlabel('p_0')
ylabel('p_1')
plot_bari()

subplot(1,2,2)
pMatching_ratio = log(1-pp1)./log(1-pp0) ./ (pp1./pp0);
pcolor_with_contour(tril(pp0), tril(pp1), tril(pMatching_ratio), gca,[1,1.2,1.5,2,2.5])
colormap('jet');  caxis([0,2])
xlabel('p_0')
ylabel('p_1')

title('pMatching index = m*/(p_1/p_0)')
SetFigure(17)
set(findall(gcf,'type','axes'),'ytick',get(gca,'xtick'))
plot_bari()
%}

%%

function [m_star, n_star, p_star] = m_star_optimal(p0,p1)
    m_star = nan(size(p0));
    n_star = m_star;
    p_star = m_star;
    
    m = 1:1000;
    n = 1:100;
    [mm, nn] = meshgrid(m, n);

    for x = 1:size(p0,1)
        for y = 1:size(p0,2)
            if p1(x,y) >= p0(x,y) && p0(x,y)>0
%                 temp = (1-p1(x,y).^2-(1-p0(x,y)).^m)./m;
                p_temp =  (2+(mm-1).*p1(x,y)+(nn-1).*p0(x,y)-(1-p1(x,y)).^(nn+1)-(1-p0(x,y)).^(mm+1))./(mm+nn);
                [p_star_this,I] = max(p_temp(:));
                p_star(x,y) = p_star_this;
                m_star(x,y) = mm(I);
                n_star(x,y) = nn(I);
             end
        end
    end
end

function pcolor_with_contour(x,y,z,ax,contour_lines)
axes(ax); cla
h = pcolor(x, y, z); view(0,90); hold on
set(h,'edgecolor','none'); h.ZData = h.CData;

if ~exist('contour_lines','var')
    [c,h]=contour(x, y, z, 'linecolor','k');
    h.ContourZLevel = max(zlim);
    clabel(c,h,'fontsize',15)
else
    if ~isempty(contour_lines)
        [c,h]=contour(x, y, z,contour_lines, 'linecolor','k');
        h.ContourZLevel = max(zlim);
        clabel(c,h,'fontsize',15)
    end
end

axis square; colorbar

end

function plot_bari()
pairs = [[.4,.05];[.3857,.0643];[.3375,.1125];[.225,.225]];
hold on;
for p = 4:-1:1
    plot3(pairs(p,2),pairs(p,1),max(zlim),'o','markersize',10,'markeredgecolor','k','linew',1.5,'markerfacecolor',[0.9100 0.4100 0.1700])
end
plot3([0,0.45],[0.45,0],[max(zlim) max(zlim)],'--','linew',2,'color',[0.9100 0.4100 0.1700]);
end

%%
%{
%%
mm = 1000; nn = 3;
((mm-1)*p1 + p_bait(p0, mm+1) + (nn-1)*p0 + p_bait(p1, nn+1))./(mm+nn)



%% 
syms m n p0 p1;
diff_p = diff(((m-1)*p1+(n-1)*p0+2-(1-p1).^(n+1)-(1-p0)^(m+1))/(m+n),n);

p_m_n = ((m-1)*p1+(n-1)*p0+2-(1-p1).^(n+1)-(1-p0).^(m+1))/(m+n);
p_m_np1 = subs(p_m_n, n, n+1);
% pretty(collect(simplifyFraction((subs(p_m_n,n,1) - subs(p_m_n,n,2))*(m+1)*(m+2)),[(1-p0)^m (1-p1)^n p0 p1]))
pretty(collect(simplifyFraction(p_m_n - p_m_np1)*(m+n)*(m+n+1),[(1-p0)^m (1-p1)^n p0 p1]))

subs(p_m_n - p_m_np1, [m n p0 p1], [100,1,0.1,0.2])

%% 
N = 100000;
ps = rand(2,N);
p0s = 0.1;% min(ps,[],1);
p1s = 0.2;%max(ps,[],1);
m = ceil(rand(1,N)*300);
n = ceil(rand(1,N)*300);

deter = 2-(1-p0s).^(m+1) - (1-p1s).^n.*p1s.*(1+(m+n).*(1-p1s)) - (m+1).*p0s + (m-1).*p1s; % > 0
% deter = (1-p0s).^(m+1) + (1-p1s).^n.*p1s.*(1+(m+n).*(1-p1s)) + (p0s + p1s) - m.* (p1s-p0s); % < 2 
min(deter)
% max((1-p0s).^(m+1) + (1-p1s).^n.*p1s.*(1+(m+n).*(1-p1s)) + (p0s + p1s) - m.* (p1s-p0s))  % < 2 !!! We can do this hahaha!!! ????????????????

% max(p1s.^m + p1s.^(n-1).*(1-p1s).*(1+(m+n).*p1s))
% max((1-p1s).^n.*p1s.*(1+(m+n).*(1-p1s)) - m.* p1s + p1s)
% hist(deter)


%}