import shogun as sg
import numpy as np
from scipy import stats


def mmd_test(Sample1, Sample2):
    for i in range(Sample1.shape[1]):
        x = Sample1[:, i]
        y = Sample2[:, i]

        feat_p = sg.RealFeatures(x.reshape(1, len(x)))
        feat_q = sg.RealFeatures(y.reshape(1, len(y)))

        # choose kernel for testing. Here: Gaussian
        kernel_width = 1
        kernel = sg.GaussianKernel(10, kernel_width)

        # create mmd instance of test-statistic
        mmd = sg.QuadraticTimeMMD()
        mmd.set_kernel(kernel)
        mmd.set_p(feat_p)
        mmd.set_q(feat_q)

        # compute biased and unbiased test statistic (default is unbiased)
        mmd.set_statistic_type(sg.ST_UNBIASED_FULL)
        statistic = mmd.compute_statistic()

    return statistic


def t_test(Sample1, Sample2):
    t = 0
    p_value = 0
    for i in range(Sample1.shape[1]):
        s1 = Sample1[i, :]
        s2 = Sample2[i, :]

        Levene, p1 = stats.levene(s1, s2)
        if Levene < 0.05:
            para = 'False'
        else:
            para = 'True'
        t, p_value = stats.ttest_ind(s1, s2, equal_var=para)
    return t, p_value





