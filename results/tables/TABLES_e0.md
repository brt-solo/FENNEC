# E0 artefact checks

## E0a: Sortability (0.5 = variance carries no order information; R2-sortability ~0 = none)

Raw = gCastle data before standardization. Both generators are standardized before any method sees them.

|                                        |   varsortability (raw) |   varsortability |   R2-sortability |
|:---------------------------------------|-----------------------:|-----------------:|-----------------:|
| ('FENNEC benchmark', 'ER', 'additive') |                nan     |            0.500 |            0.195 |
| ('FENNEC benchmark', 'ER', 'linear')   |                nan     |            0.500 |            0.084 |
| ('FENNEC benchmark', 'ER', 'mixed')    |                nan     |            0.500 |            0.147 |
| ('FENNEC benchmark', 'ER', 'mlp')      |                nan     |            0.500 |            0.136 |
| ('gCastle', 'ER', 'gp')                |                  0.879 |            0.500 |           -0.239 |
| ('gCastle', 'ER', 'mim')               |                  0.809 |            0.500 |           -0.138 |
| ('gCastle', 'ER', 'mlp')               |                  0.769 |            0.500 |           -0.186 |
| ('gCastle', 'ER', 'quadratic')         |                  0.835 |            0.500 |            0.219 |
| ('gCastle', 'SF', 'gp')                |                  0.873 |            0.500 |           -0.019 |
| ('gCastle', 'SF', 'mim')               |                  0.804 |            0.500 |           -0.001 |
| ('gCastle', 'SF', 'mlp')               |                  0.784 |            0.500 |           -0.058 |
| ('gCastle', 'SF', 'quadratic')         |                  0.822 |            0.500 |            0.229 |

## E0b: Penalty audit: FENNEC skeleton F1 / SHD by BIC multiplier, exploratory vs held-out seeds



|   mult |   ('skel_f1', 'exploratory seeds 0-4') |   ('skel_f1', 'held-out seeds 5-9') |   ('shd', 'exploratory seeds 0-4') |   ('shd', 'held-out seeds 5-9') |   ('datasets', 'exploratory seeds 0-4') |   ('datasets', 'held-out seeds 5-9') |
|-------:|---------------------------------------:|------------------------------------:|-----------------------------------:|--------------------------------:|----------------------------------------:|-------------------------------------:|
|      1 |                                  0.762 |                               0.751 |                             10.113 |                          11.850 |                                  80.000 |                               80.000 |
|      2 |                                  0.836 |                               0.823 |                              6.975 |                           8.213 |                                  80.000 |                               80.000 |
|      4 |                                  0.903 |                               0.877 |                              4.312 |                           5.550 |                                  80.000 |                               80.000 |
|      8 |                                  0.901 |                               0.866 |                              3.837 |                           5.400 |                                  80.000 |                               80.000 |

## E0b2: Best multiplier per (mechanism, n) cell, by seed split



|                    |   exploratory seeds 0-4 |   held-out seeds 5-9 |
|:-------------------|------------------------:|---------------------:|
| ('additive', 200)  |                       4 |                    4 |
| ('additive', 1000) |                       8 |                    8 |
| ('linear', 200)    |                       8 |                    4 |
| ('linear', 1000)   |                       4 |                    4 |
| ('mixed', 200)     |                       4 |                    4 |
| ('mixed', 1000)    |                       8 |                    8 |
| ('mlp', 200)       |                       4 |                    4 |
| ('mlp', 1000)      |                       8 |                    4 |

## E0b3: Held-out seeds: paired skeleton-F1 difference, 4x minus alternative (95% CI)



| comparison             |   skel F1 diff |     lo |    hi |
|:-----------------------|---------------:|-------:|------:|
| 4x minus 1x (held-out) |          0.125 |  0.106 | 0.144 |
| 4x minus 2x (held-out) |          0.054 |  0.040 | 0.068 |
| 4x minus 8x (held-out) |          0.011 | -0.002 | 0.025 |

## E0c: Outside generator (gCastle, standardized): mean over all datasets



|                                   |   skel_f1 |   orient_recall |    shd |   seconds |
|:----------------------------------|----------:|----------------:|-------:|----------:|
| CAM                               |     0.714 |           0.528 | 14.705 |    24.425 |
| NoGAM                             |     0.678 |           0.449 | 18.066 |    28.083 |
| SCORE                             |     0.669 |           0.420 | 18.738 |     9.902 |
| GES (BIC)                         |     0.628 |           0.329 | 24.956 |     0.511 |
| PC                                |     0.603 |           0.236 | 24.781 |     0.185 |
| var_sortnregress                  |     0.574 |           0.279 | 34.894 |     0.042 |
| FENNEC (XGBoost CV score, 4x BIC) |     0.566 |           0.410 | 18.078 |     4.076 |
| R²-sortnregress                   |     0.555 |           0.256 | 39.278 |     0.067 |

## E0c2: Outside generator: skeleton F1 by graph type, mechanism and n



|                                   |   ('ER', 'gp', 200) |   ('ER', 'gp', 1000) |   ('ER', 'mim', 200) |   ('ER', 'mim', 1000) |   ('ER', 'mlp', 200) |   ('ER', 'mlp', 1000) |   ('ER', 'quadratic', 200) |   ('ER', 'quadratic', 1000) |   ('SF', 'gp', 200) |   ('SF', 'gp', 1000) |   ('SF', 'mim', 200) |   ('SF', 'mim', 1000) |   ('SF', 'mlp', 200) |   ('SF', 'mlp', 1000) |   ('SF', 'quadratic', 200) |   ('SF', 'quadratic', 1000) |
|:----------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|---------------------:|----------------------:|---------------------------:|----------------------------:|--------------------:|---------------------:|---------------------:|----------------------:|---------------------:|----------------------:|---------------------------:|----------------------------:|
| CAM                               |               0.587 |                0.836 |                0.674 |                 0.905 |                0.692 |                 0.796 |                      0.566 |                       0.688 |               0.549 |                0.810 |                0.682 |                 0.904 |                0.702 |                 0.818 |                      0.538 |                       0.675 |
| NoGAM                             |               0.561 |                0.817 |                0.648 |                 0.918 |                0.658 |                 0.733 |                      0.489 |                       0.648 |               0.502 |                0.788 |                0.654 |                 0.897 |                0.665 |                 0.772 |                      0.467 |                       0.632 |
| SCORE                             |               0.530 |                0.803 |                0.649 |                 0.923 |                0.643 |                 0.741 |                      0.476 |                       0.639 |               0.493 |                0.775 |                0.635 |                 0.901 |                0.669 |                 0.762 |                      0.460 |                       0.610 |
| GES (BIC)                         |               0.488 |                0.657 |                0.740 |                 0.850 |                0.650 |                 0.674 |                      0.512 |                       0.541 |               0.488 |                0.630 |                0.693 |                 0.818 |                0.651 |                 0.692 |                      0.434 |                       0.522 |
| PC                                |               0.477 |                0.663 |                0.705 |                 0.847 |                0.581 |                 0.651 |                      0.475 |                       0.536 |               0.453 |                0.640 |                0.643 |                 0.816 |                0.572 |                 0.663 |                      0.412 |                       0.511 |
| var_sortnregress                  |               0.479 |                0.638 |                0.697 |                 0.726 |                0.579 |                 0.578 |                      0.491 |                       0.525 |               0.450 |                0.597 |                0.665 |                 0.705 |                0.556 |                 0.542 |                      0.442 |                       0.517 |
| FENNEC (XGBoost CV score, 4x BIC) |               0.236 |                0.708 |                0.387 |                 0.848 |                0.598 |                 0.823 |                      0.295 |                       0.565 |               0.238 |                0.674 |                0.439 |                 0.839 |                0.646 |                 0.831 |                      0.350 |                       0.572 |
| R²-sortnregress                   |               0.451 |                0.622 |                0.676 |                 0.666 |                0.563 |                 0.557 |                      0.505 |                       0.522 |               0.447 |                0.588 |                0.628 |                 0.649 |                0.526 |                 0.531 |                      0.439 |                       0.506 |

## E0d: Sort-and-regress artefact baselines vs FENNEC



|                                                           |   skel F1 |   oriented recall |   datasets |
|:----------------------------------------------------------|----------:|------------------:|-----------:|
| ('FENNEC benchmark', 'R²-sortnregress')                   |     0.543 |             0.321 |    160.000 |
| ('FENNEC benchmark', 'var_sortnregress')                  |     0.621 |             0.430 |    160.000 |
| ('FENNEC benchmark', 'FENNEC (XGBoost CV score, 4x BIC)') |     0.890 |             0.741 |    160.000 |
| ('gCastle', 'R²-sortnregress')                            |     0.555 |             0.256 |    320.000 |
| ('gCastle', 'var_sortnregress')                           |     0.574 |             0.279 |    320.000 |
| ('gCastle', 'FENNEC (XGBoost CV score, 4x BIC)')          |     0.566 |             0.410 |    320.000 |
