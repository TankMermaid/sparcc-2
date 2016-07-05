'''
Created on Jun 24, 2012

@author: jonathanfriedman 
'''

from pandas import DataFrame as DF
from core_methods import _get_axis
import numpy as np

#-------------------------------------------------------------------------------
# Diversity methods   

def sample_diversity(frame, indices='Hill_1', **kwargs):
    '''
    Calculate diversity indices of all rows. 
    See :func:`pysurvey.analysis.diversity.sample_diversity` for detailed documentation.
    
    Note that some methods work only with unnormalized data!
        
    Parameters
    ----------
    frame : DataFrame
        matrix of observed counts/fractions. 
        Columns are treated as components and rows as samples. 
        Some methods would also work with proportions and/or incidence data.
    indices : str/iterable (default 'Hill_1')
        list of indices of diversity to use. 
        Valid values:
        
        - Hill_n   = Hill number of order n
        - Richness = Sames as Hill_0.
        - Shannon  = Shannon entropy. Same as Reyni_1.
        - Renyi_n  = Reyni enropy of order n. Same as log(Hill_n). Reyni_1 is the Shannon entropy.
        - Simpson  = Simpson's index of diversity. Same as 1/Hill_2.
        - Simpson_Inv = The inverse of simpson's index. Same as Hill_2.
    
    methods : str/iterable (default 'ML')
        list of same length as indices detailing the methods used to calculate each index.
        Default is ML for all methods.
    
    Returns
    -------
    ne_ff : frame
        DataFrame of estimated diversity indices. 
        Columns are indices and rows are samples.
    '''
    from pysurvey.analysis.diversity import sample_diversity as sd
    if isinstance(indices, str): indices  = [indices]
    d, indices, methods = sd(frame.values, indices=indices, **kwargs)    
    indices_str = [i +'.' + m for i,m in zip(indices, methods) ]
    n_eff       = DF(d, index=frame.index, columns=indices_str)     
    return n_eff

#-------------------------------------------------------------------------------
# Pairwise similarity methods   

def dist_mat(frame, metric='euclidean', axis=0, **kwargs):
    '''
    Calculate the (symmetric) pairwise distnce metric.
    :func:`scipy.cluster.hierarchy.distance.pdist` is used to do the actual
    distance computation.
    
    Parameters
    ----------
    metric : str/callable
        Any metric supported by :func:`scipy.cluster.hierarchy.distance.pdist`
        Additional supported metrics:
        
        - JS - Jensen-Shannon divergence. 
          Information theoretic measure of distance between 
          distributions (or any sets of numbers who's sum is 1). 
          Gives each taxa a weight proportional to its relative abundance.               
        - JSsqrt - the square-root of Jensen-Shannon divergence. 
          Upholds the triangular inequality, and is therefore a 
          true metric.
        - Morisita - the Morisita-Horn dissimilarity index. 
          Gives more weight to more abundant taxa. 
    
    axis :  {0 (default)| 1}
    
        - 0 - calculate distance between rows.
        - 1 - calculate distance between columns.
    
    kwargs : 
        Additional keyword arguments to be passed to 
        :func:`scipy.cluster.hierarchy.distance.pdist`
    
    Returns
    -------
    D : frame
        DataFrame of symmetric pairwise distances.
        Labels are the rows/column labels of the input frame. 
    '''
    import pysurvey.util.distances as distances
    axis = _get_axis(axis)
    if   axis == 0: data = frame
    elif axis == 1: data = frame.T
    mat = data.values
    row_labels = data.index
    D_mat = distances.pdist(mat, metric, **kwargs)
    D = DF(D_mat, index=row_labels, columns=row_labels)
    return D

def basis_corr(frame, algo='SparCC', **kwargs):
    '''
    Compute correlations between all columns of a counts frame.
    This is a wrapper around pysurvey.analysis.basis_correlations.main
        
    Parameters
    ----------
    counts : array_like
        2D array of counts. Columns are components, rows are samples. 
    method : str {SparCC (default)| clr| pearson| spearman| kendall}
        The algorithm to use for computing correlation.

    Returns
    -------
    cor_med: frame
        Estimated correlation matrix.
        Labels are column labels of input frame.
    cov_med: frame/None
        If method in {SparCC, clr} : Estimated covariance matrix.
        Labels are column labels of input frame. 
        Otherwise: None.
              
    =======   ============ =======   ================================================
    kwarg     Accepts      Default   Desctiption
    =======   ============ =======   ================================================
    iter      int          20        number of estimation iteration to average over.
    oprint    bool         True      print iteration progress?
    th        0<th<1       0.1       exclusion threshold for SparCC.
    xiter     int          10        number of exclusion iterations for sparcc.
    norm      str          dirichlet method used to normalize the counts to fractions.
    log       bool         True      log-transform fraction? used if method ~= SparCC/CLR
    =======   ============ ========= ================================================
    '''
    import pysurvey.analysis.basis_correlations.get_correlations as get_cor
    comps  = frame.columns
    cor_med, cov_med = get_cor.main(frame, algo=algo, **kwargs)
    cor = DF(cor_med, index=comps, columns=comps)
    if cov_med is None:
        cov = None
    else:
        cov  = DF(cov_med, index=comps, columns=comps)
    return cor, cov  

def correlation(frame, method='pearson', axis=0):
    '''
    Calculate the correlation between all rows/cols.
    Return frames of correlation values and p-values.
    
    Parameters
    ----------
    frame : DataFrame
        Frame containing data.
    method : {pearson (default) | spearman | kendall}
        Type of correlations to be computed
    axis : {0, 1}
        - 0 - Compute correlation between columns
        - 1 - Compute correlation between rows
    
    Returns
    -------
    c : frame
        DataFrame of symmetric pairwise correlation coefficients.
        Labels are the rows/column labels of the input frame.
    p : frame
        DataFrame of p-values associated with correlation values.
        Labels are the rows/column labels of the input frame.
    ''' 
    import scipy.stats as stats
    axis = _get_axis(axis)
    method = method.lower()
    if method not in set(['pearson', 'kendall', 'spearman']): 
        raise ValueError('Correlation of method %s is not supported.' %method)
    if method == 'spearman' : 
        c_mat, p_mat = stats.spearmanr(frame.values, axis=axis)
        if not np.shape(c_mat):
            c_mat = np.array([[1, c_mat],[c_mat,1]])
            p_mat = np.array([[1, p_mat],[p_mat,1]])
        labels = frame._get_axis(1-axis)
        c = DF(c_mat, index=labels, columns=labels)
        p = DF(p_mat, index=labels, columns=labels)
    else:
        if   method == 'pearson': corr_fun = stats.pearsonr
        elif method == 'kendall': corr_fun = stats.kendalltau
        if   axis == 0: data = frame.T
        elif axis == 1: data = frame
        mat = data.values
        row_labels = data.index
        n = len(row_labels)
        c_mat = np.zeros((n, n))
        p_mat = np.zeros((n, n))
        for i in xrange(n):
            for j in xrange(i, n):
                if i == j: 
                    c_mat[i][i] = 1
                    p_mat[i][i] = 1
                    continue
                c_temp, p_temp = corr_fun(mat[i, :], mat[j, :])
                c_mat[i][j] = c_temp
                c_mat[j][i] = c_temp
                p_mat[i][j] = p_temp
                p_mat[j][i] = p_temp
        c = DF(c_mat, index=row_labels, columns=row_labels)
        p = DF(p_mat, index=row_labels, columns=row_labels)
    return c, p

#-------------------------------------------------------------------------------
# Clustering/dimension reduction
       
def PCoA(frame, metric=None, **kwargs):
    '''
    Do PCoA (= metric Multidimensional scaling) of frame.
    Treat rows as samples and columns as coordinates. 
    Eigenvalues and coordinates are order by eigenvalue in 
    descending order.
    
    Parameters
    ----------
    metric : str/None 
        Metric used to calculate the distance matrix of frame.
        If None is given, treat input frame a distance matrix.
    **kwargs :
        Additional kwargs for DM.dist_mat.
        
    Returns
    -------
    points_sorted : DataFrame
        Rows are samples and columns are principle axis.
    eigs_sorted : Series 
        Eigenvectors of each axis.
    '''
    from cogent.cluster.metric_scaling import principal_coordinates_analysis
    from scipy.spatial.distance import is_valid_dm
    from pandas import Series
    
    if metric is None:
        D = frame
    else:
        D = dist_mat(frame, metric=metric, **kwargs)
    if not is_valid_dm(D):
        raise ValueError, 'Input is not a valid distance matrix.'   
    n = len(D)
    x,eigs  = principal_coordinates_analysis(D.values)
    rlabels = ['PC%d' %i for i in xrange(1,n+1)][::-1]
    eigs = Series(eigs,index = rlabels)
    eigs_sorted = eigs[::-1]
    points = DF(x, index=rlabels, columns=D.columns)
    points_sorted = points.reindex(index=eigs_sorted.index)
    return points_sorted.T, eigs_sorted

#-------------------------------------------------------------------------------
# Feature selection methods
def discriminating_components(frame1, frame2, method='U'):
    '''
    For each component, compare its distribution across two groups
    using a statistical test (see methods).
    
    TODO: extend to multiple sample groups
    TODO: use pandas groupby functionality 
    
    Parameters
    ----------
    frame1/2 : DataFrame
        DataFrames to be compared (e.g. treatment and control).
        Must have identical column labels.
    method : {'U' (default)| 'Fisher'}
        Method/statistical test used to score discriminating components.
        'U' : Mann-Whitney U test.
        'Fisher' : Fisher's exact test (requires count data). 
    Returns
    -------
    DataFrame whose rows correspond to the columns of the input frames,
    and columns are the medians of the two frames, and the test p-value.
    Rows are sorted according to p-value.    
    '''
    from warnings import warn
    n1,m = np.shape(frame1)
    method = method.lower()
    if method == 'u':
        import scipy.stats as stats
        out = DF(np.zeros((m,3)), index=list(frame1.columns), columns=['Median1','Median2', 'p-val'])
        for id,vals1 in frame1.iteritems():
            vals2 = frame2[id]
            try:
                s, p = stats.mannwhitneyu(vals1, vals2)
            except ValueError:
                s = np.nan
                p = 1.
                warn('All numbers are identical in amannwhitneyu. Setting p-val=1.')
            out['Median1'][id] = vals1.median()
            out['Median2'][id] = vals2.median()
            out['p-val'][id] = p
    elif method == 'fisher':
        from pysurvey.util.R_utilities import fisher_test
        from pysurvey.core.core_methods import to_binary
        th = 0
        frame_bin  = to_binary(frame1, th)
        other_bin  = to_binary(frame2, th)
        frame_pres = frame_bin.sum()
        other_pres = other_bin.sum()
        out = DF(np.zeros((m,3)), index=list(frame1.columns), columns=['Presence1','Presence2', 'p-val'])
        n2,m = np.shape(frame2)
        for id,vals1 in frame1.iteritems():
            p1 = frame_pres[id]
            p2 = other_pres[id]
            table = np.array([[p1,n1-p1],[p2,n2-p2]]) # make contingency table
            p_val = fisher_test(table)[0]
            out['Presence1'][id] = p1
            out['Presence2'][id] = p2
            out['p-val'][id] = p_val
    return out.sort(columns='p-val')

#-------------------------------------------------------------------------------
# Misc.                    
def permute_w_replacement(frame, axis=0):
    '''
    Permute the frame values across the given axis.
    Create simulated dataset were the counts of each component (column)
    in each sample (row), are randomly sampled from the all the 
    counts of that component in all samples.
    
    Parameters
    ----------
    frame : DataFrame
        Frame to permute.
    axis : {0, 1}
        - 0 - Permute row values across columns
        - 1 - Permute column values across rows    
    
    Returns
    -------
    Permuted DataFrame (new instance).
    '''
    from numpy.random import randint 
    axis = 1-_get_axis(axis)
    s = frame.shape[axis]
    fun = lambda x: x.values[randint(0,s,(1,s))][0]
    perm = frame.apply(fun, axis=axis)
    return perm


#-------------------------------------------------------------------------------
# Ecological analysis
   
def rank_abundance(frame):
    '''
    Compute the rank abundance relation of all the samples (rows).
    
    Returns
    -------
    DataFrame whose columns are ranks and 
    rows are the same as those of the input frame.
    '''
    # rank order
    fun = lambda y: sorted(y.values, reverse=True)
    ranked = frame.apply(fun,axis=1)
    # set rank labels
    from pysurvey import set_labels
    m = frame.shape[1]
    set_labels(ranked, np.arange(1,m+1), axis=1)
    return ranked

if __name__ == '__main__':
    pass
    
#    df = pandas.DF([[1,3,2],[4,6,5]], columns=['a','b','c'], index=['r1','r2'])
#    print df,'\n'
#    print df.correlation(axis=1), '\n'
#    print df.correlation(axis=0), '\n'
#    print df.correlation('spearman', axis='cols_'), '\n'
#    print df.correlation('spearman', axis='rows_')
    
#    df.set_axisName('rows','samples')
#    dft = df.T
#    print df.min(axis='samples_'), '\n'
#    print dft.min(axis='samples_'), '\n'
#    print df.T,'\n'
#    print df.sort(columns='a',axis='0')
#    print df.apply(lambda x:x['r2'], axis='rows'), '\n'
#    print df.apply(lambda x:x['r2']), '\n'
    
