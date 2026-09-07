#!/usr/bin/env python3

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import mpmath as mp
import os

matplotlib.use("Agg") 

################################################################################
#                                                                              #
#                             BFO23 & JMB24                                    #
#                                                                              #
################################################################################

def proba_edge_mult_paper(p, n):
  """
  Theorem 1 of BFO23.

  Args:
      p (float): Leakage rate.
      n (int): Number of shares.

  Returns:
      float: Advantage in the Random probing security of Mult.
  """
  return 1 - (1 - p) ** (8 * n) + 1 - (1 - math.sqrt(3 * p)) ** (n - 1)

def expr_BFO23(p, n, nb_gadgets):
  """
  Theorem 4 of BFO23.

  Args:
      p (float): Leakage rate.
      n (int): Number of shares.
      nb_gadgets (int): Number of gadgets, i.e. circuit size |C|.

  Returns:
      float: Advantage in the Random probing security of AES.
  """
  return min(1.0, nb_gadgets * (18 * p + 2 * proba_edge_mult_paper(p, n)) ** n)

def expr_JMB24(p, n):
  """
  Section 7.1 of JMB24.

  Args:
      p (float): Leakage rate.
      n (int): Number of shares.

  Returns:
      float: Advantage in the Random probing security of AES.
  """
  return 4 * (5.3 * p) ** (0.3 * n)


################################################################################
#                                                                              #
#                        Probability of adding edges.                          #
#                                                                              #
################################################################################

def proba_edge_add_copy(p, n):
  """
  Compute the probability of addition of one edge into the Leakage Diagram,
  following Section 5.2. For the formula in Section 5.2, we have here 
  t_aff = 3 and w = 1. 

  Args:
      p (float): Leakage Rate.
      n (int): Number of shares.

  Returns:
      float: The probability of adding one edge of addition/copy gadget to the
      Leakage Diagram.
  """
  return 1 - (1 - p)**3

def proba_edge_11(p, n):
  """
  Compute the probability of addition of one edge into the Leakage Diagram,
  following Section 5.2. For the formula in Section 5.2, we have here 
  t_aff = 2 and w = 1. 

  Args:
      p (float): Leakage Rate.
      n (int): Number of shares.

  Returns:
      float: The probability of adding one edge of unary gadget to the
      Leakage Diagram.
  """
  return 1 - (1 - p)**2


"""
    Computing
      1 - |S| (1-q1) (1-q2)^(n-1)
      + sum_{l=2..|S|} C(|S|,l) (-1)^l (1-q1)^l (1-q2)^( l(l-1)/2 + l(n-l) )

    with :
      q1 = 1 - (1 - p)^(6n)
      q2 = 1 - (1 - p)**3
"""

def pr_add_s_mul_edges(n, w, p, dps=200):
  """
  Compute the probability of adding |w| edge of a multiplication together.
  Following Lemma 4 of the paper, we have to compute :
      
      1 + sum_{l=1..|S|} C(|S|,l) (-1)^l (1-q1)^l (1-q2)^( l(l-1)/2 + l(n-l) )
  
  where q1 = (1 - p)**6n and q2 = (1 - p)**3.

  Since this computation involves very small quantities and an alternating sum
  of terms that may be close in magnitude, standard floating-point arithmetic
  may suffer from significant numerical errors. We therefore use the `mpmath`
  arbitrary-precision arithmetic library. The precision is controlled by the
  `dps` parameter.

  Args:
      n (int): Number of shares.
      w (float/int): Number of edges of the multiplication gadgets we add. 
      p (float): Leakage Rate.
      dps (int, optional): Number of decimal digits of precision used for
                           arbitrary-precision arithmetic. Defaults to 200.

  Returns:
      float: Probability of adding together |w| edges of the multiplication
      gadget into the Leakage Diagrams.
  """

  mp.mp.dps = dps

  n = int(n)
  s = int(w)

  p = mp.mpf(p)
  one = mp.mpf(1)

  log_one_minus_q1 = 6 * n * mp.log1p(-p)
  log_one_minus_q2 = 3 * mp.log1p(-p)

  terms = [one]
  # Sum over l = 1, ..., |S|
  for l in range(1, s + 1):
        expo = mp.mpf((l * (l - 1)) // 2 + l * (n - l))
        log_term_l = (
            mp.log(mp.binomial(s, l)) + l * log_one_minus_q1 + expo * log_one_minus_q2
        )
        terms.append((-1)**l * mp.exp(log_term_l))

  return mp.fsum(terms)


def pr_add_s_mul_edges_cache(n, p, dps):
  """
  Compute |pr_add_s_mul_edges| for all w in range (1, w) and store it.

  Args:
      n (int): Number of Shares.
      p (float): Leakage Rate.
      dps (int): Number of decimal digits of precision used for
                 arbitrary-precision arithmetic.

  Returns:
      list[float]: List of probability. The i-th index corresponds to the
      probability of adding i edges together of the multiplication gadget into
      the Leakage Diagram.
  """
  cache_res = [0]
  for s_card in range(1, n + 1):
    cache_res.append(pr_add_s_mul_edges(n, s_card, p, dps))
  return cache_res

def integer_partitions(n: int, max_part: int | None = None):
  """
  Generates all partitions of |n| into positive parts, in decreasing order.

  Args:
      n (int): The integer.
      max_part (int | None, optional): Maximal number of partitions we are 
                                       allowed. Defaults to None.

  Yields:
      tuple[int, ...]: A new tuple of integer partitions of n. 
  """
  if n < 0:
    return
  if max_part is None or max_part > n:
    max_part = n

  if n == 0:
    yield ()
    return

  for k in range(max_part, 0, -1):
    for rest in integer_partitions(n - k, k):
      yield (k,) + rest


def max_proba_mult(n, delta, p, cache):
  """
  We compute the maximale probability from which |delta| multiplication gadgets
  edges are added to the leakage diagrams. In particular, these edges are added
  into different plateau of the Leakage Diagram. This is why we need to compute
  all the integer partitions of |delta| and take the maximale probability among
  each integer partition.

  Args:
      n (int): Number of shares.
      delta (int): Number of edges of the multiplication gadget that leaks (on
      different plateau of the Leakage Diagram).
      p (int): Leakage Rate.
      cache (list[float]): List of probabilities, returned from 
                          |pr_add_s_mul_edges_cache|.

  Returns:
      float: Maximale probabilities that |delta| edges of multiplication gadgets
      are added to the Leakage Diagrams.
  """
  max_proba = 0
  for int_part in integer_partitions(delta, n - 1):

    proba_int_part = 1
    for card in int_part:
      proba_int_part *= cache[card]

    max_proba = max(proba_int_part, max_proba)

  return max_proba

################################################################################
#                                                                              #
#                              Utils                                           #
#                                                                              #
################################################################################

def ln_one_minus_val(val, n):
  """
  Compute ln(1  - val**n) in a stable way.

  Args:
      val (float): Probability in the interval [0, 1].
      n (int): Exponent.

  Returns:
      float: ln(1 - val**n)
  """

  n_mp = mp.mpf(n)

  # log(val) can be instable with log1p if val ≈ 1
  if abs(val - 1) < mp.mpf("1e-10"):
    ln_val = mp.log1p(val - 1)
  else:
    ln_val = mp.log(val)

  ln_x = n_mp * ln_val

  if ln_x > -mp.mpf("1e-10"):
    return mp.log(-mp.expm1(ln_x))
  else:
    return mp.log1p(-mp.exp(ln_x))


def alpha_sum(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, fadd, fmult, fun, 
              fcopy, t, cache):
  """
  Compute the sum :

  Sum_{\alpha \in {add, copy, un, mult}} f_{alpha}*p_{alpha}^{exp}*|G_{alpha}|

  This sum is useful to compute the term c_{n + 2} as we can see in Lemma 7 of
  the article. Indeed, it is used to compute the function \Phi_\gamma(t),
  \Psi(t) and \Theta(t) of Lemma 7.

  Args:
      p (float): Leakage rate.
      n (int): Number of shares.
      nb_gun (int): Number of unary gadgets (i.e. multiplication by a constant gadget.)
      nb_gcopy (int): Number of copy gadgets.
      nb_gadd (int): Number of addition gadgets.
      nb_gmult (int): Number of multiplication gadgets.
      fadd (int): Additional coefficient for addition gadget.
      fmult (int): Additional coefficient for multiplication gadget.
      fun (int): Additional coefficient for unary gadget.
      fcopy (int): Additional coefficient for copy gadget.
      t (int): Number of edges added to the Leakage Diagram.
      cache (list[float]): List of probabilities, returned from 
                      |pr_add_s_mul_edges_cache|.

  Returns:
      float: the sum described above.
  """


  p_add = proba_edge_add_copy(p, n)
  p_unary = proba_edge_11(p, n)

  return (
         fadd * (p_add**t) * nb_gadd
         + fmult * cache[t] * nb_gmult
         + fun * (p_unary**t) * nb_gun
         + fcopy * (p_add**t) * nb_gcopy
  )


################################################################################
#                                                                              # 
#                           Probabilities of Theorem 2                         #
#                                                                              #
################################################################################

def expr_cn(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, cache):
  """
  Compute the probability that the Random Probing Leakage Diagram contain at
  least an orbit of size n. This is done following Lemma 5 of the article.

  Args:
      p (float): Leakage rate.
      n (int): Number of shares.
      nb_gun (int): Number of unary gadgets.
      nb_gcopy (int): Number of copy gadgets.
      nb_gadd (int): Number of addition gadgets.
      nb_gmult (int): Number of multiplication gadgets.
      cache (list[float]): List of probabilities, returned from 
                      |pr_add_s_mul_edges_cache|.

  Returns:
      float: Probability that the Random Probing Leakage Diagram contain at
              least an orbit of size n.
  """

  # 1) Computation of p_add and p_un, where p_add = p_copy.
  pa = proba_edge_add_copy(p, n)
  pun = proba_edge_11(p, n)
  nb_gadgets_add_copy = nb_gcopy + nb_gadd

  # 2) Computation of the log probability to add n edges of the multiplication.
  log_pm_pow_n = mp.log1p(-cache[n])

  # 3) log(Product of the terms).
  ln_prod = (
    nb_gadgets_add_copy * ln_one_minus_val(pa, n)
    + nb_gun * ln_one_minus_val(pun, n)
    + nb_gmult * log_pm_pow_n
  )

  if ln_prod == mp.ninf:
    print("Precision Error")
    exit()
  else:
    # 4) Exponential to return cn.
    cn = -mp.expm1(ln_prod)

  return cn

def expr_cn1(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, cache):
  """
  Compute the probability that the Random Probing Leakage Diagram contain at
  least an orbit of size n + 1. This is done following Lemma 6 of the article.

  Args:
      p (float): Leakage rate.
      n (int): Number of shares.
      nb_gun (int): Number of unary gadgets.
      nb_gcopy (int): Number of copy gadgets.
      nb_gadd (int): Number of addition gadgets.
      nb_gmult (int): Number of multiplication gadgets.
      cache (list[float]): List of probabilities, returned from 
                      |pr_add_s_mul_edges_cache|.

  Returns:
      float: Probability that the Random Probing Leakage Diagram contain at
              least an orbit of size n + 1.
  """

  pa = proba_edge_add_copy(p, n)
  pun = proba_edge_11(p, n)
  sum_rb = 0

  # 1) Upward Orbit formula.
  for b in range(0, n):
    for r in range(b, n + b - 1):
      tmp = (
        cache[n - 1 - r + b] * nb_gmult
        + (pa ** (n - 1 - r + b)) * nb_gadd
        + (pun ** (n - 1 - r + b)) * nb_gun
        + 2 * (pa ** (n - 1 - r + b)) * nb_gcopy
      )
      tmp2 = pr_add_s_mul_edges(n, r - b, p)
      sum_rb += tmp * tmp2
  term1 = sum_rb * 3 * (p**2)

  # 2) Downward Orbit formula.
  term2 = (
    3 * (p**2) * n * (
      2 * nb_gmult * cache[n - 1]
      + 2 * nb_gadd * (pa ** (n - 1))
      + nb_gcopy * (pa ** (n - 1))
      + nb_gun * (pun ** (n - 1))
    )
  )

  return term1 + term2

def expr_cn2(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, cache, max_proba_mult_cache):
  """
  Compute the probability that the Random Probing Leakage Diagram contain at
  least an orbit of size n + 2. This is done following Lemma 7 of the article.

  Args:
      p (float): Leakage rate.
      n (int): Number of shares.
      nb_gun (int): Number of unary gadgets.
      nb_gcopy (int): Number of copy gadgets.
      nb_gadd (int): Number of addition gadgets.
      nb_gmult (int): Number of multiplication gadgets.
      max_proba_mult_cache (list[float]): List of n probabilities, containing 
        for each i from 1 to n, the probability returned by 
        |max_proba_mult(n, i, p, cache)|.

  Returns:
      float: Probability that the Random Probing Leakage Diagram contain at
              least an orbit of size n + 2.
  """
    

  comb = math.comb
  pa = proba_edge_add_copy(p, n)

  # Sum computed in
  # 1.Orbits going through two b-edges of the upward refresh gadget.
  # 2.Orbits going through two r-edges of the upward refresh gadget.
  sum_term1 = 0
  for b in range(0, n - 1):
    for bp in range(b + 1, n):
      delta = bp - b
      sum_term1 += cache[delta] * alpha_sum(
        p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 1, 1, 1, 2, n - delta, cache
      )
      sum_term1 += cache[n - delta] * alpha_sum(
        p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 1, 1, 1, 2, delta, cache
      )

  # Sum computed in
  # 3.Orbits reaching two gadgets upwards.
  sum_term3 = 0
  for delta in range(1, n - 1):
    for b in range(1, n - delta):
      sum_term3 += (
        (n - delta - b) * (max_proba_mult_cache[n - 2 - delta])
        * alpha_sum(
            p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 1, 1, 1, 2, delta, cache
          )
      )

  # Sum computed in
  # 5.Orbits reaching one gadget upward
  sum_term5_1 = 0
  for r0 in range(0, n):
    for b0 in range(r0 + 1, n):
      for r1 in range(b0, n):
        for b1 in range(r1 + 1, n):
          if (b0, b1) == (r0 + 1, r1 + 1):
            continue
          delta = (b0 - r0) + (b1 - r1) - 2
          sum_term5_1 += (cache[n - 2 - delta]) * alpha_sum(
            p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 1, 1, 1, 4, delta, cache)

  tmp_term5_1 = ((3 * (p**2)) ** 2) * sum_term5_1

  sum_term5_2 = 0
  for b0 in range(0, n):
    for r0 in range(b0, n):
      for b1 in range(r0 + 1, n):
        for r1 in range(b1, n):
          if (b0, b1, r1) == (0, r0 + 1, n - 1):
            continue

          delta = (r0 - b0) + (r1 - b1)
          sum_term5_2 += (cache[delta]) * alpha_sum(
            p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 1, 1, 1, 4,
            n - 2 - delta, cache)

  tmp_term5_2 = ((3 * (p**2)) ** 2) * sum_term5_2

  # Sum computed in
  # 7.Orbits reaching one gadget upward and one gadget downward.
  tmp_term7_1 = (
    n * (n - 1) * ((3 * p**2) ** 2)
    * alpha_sum(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 2, 2, 1, 2, n - 2, cache)
  )
  sum_term7 = 0
  for b0 in range(0, n - 1):
    for r0 in range(b0 + 1, n - 1):
      if (b0, r0) == (0, n - 2):
        continue
      d = r0 - b0
      sum_term7 += (cache[d]) * alpha_sum(
        p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 2, 2, 1, 2, n - 2 - d, cache)

  tmp_term7_2 = n * ((3 * p**2) ** 2) * sum_term7

  # Sum computed in
  # 8. Orbits reaching one copy gadget upward and its second output.
  sum_term8 = 0
  for d in range(3, n):
    sum_term8 += cache[d - 2] * alpha_sum(
      p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 2, 2, 1, 1, n - d, cache)

  term1 = (p**2) * sum_term1
  term2 = ((3 * p) ** 2) * sum_term1
  term3 = sum_term3 * 2 * n * ((3 * p**2) ** 2)
  term4 = (
    n * ((3 * p**2) ** 2) * 2
    * alpha_sum(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 2, 2, 1, 1, n - 2, cache)
  )
  term5 = tmp_term5_1 + tmp_term5_2
  term6 = (
        comb(n, 2) * ((3 * p**2) ** 2)
        * alpha_sum(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, 1, 1, 1, 2, n - 2, cache)
        + (n**2) * ((3 * p**2) ** 2) * (pa ** (n - 2)) * nb_gcopy
  )
  term7 = tmp_term7_1 + tmp_term7_2
  term8 = sum_term8 * (3 * p**2) ** 2 * n

  return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8


def expr_cn3(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, beta, cache, max_proba_mult_cache):
  """
  Compute the probability that the Random Probing Leakage Diagram contain at
  least an orbit of size  at least n + 3. This is done following Lemma 8 of 
  the article.

  Args:
      p (float): Leakage rate.
      n (int): Number of shares.
      nb_gun (int): Number of unary gadgets.
      nb_gcopy (int): Number of copy gadgets.
      nb_gadd (int): Number of addition gadgets.
      nb_gmult (int): Number of multiplication gadgets.
      beta(int) : Constant, adjusted for more precision.
      cache(list[float]) : List of probabilities, returned from 
                           |pr_add_s_mul_edges_cache|.
      max_proba_mult_cache (list[float]): List of n probabilities, containing 
        for each i from 1 to n, the probability returned by 
        |max_proba_mult(n, i, p, cache)|.

  Returns:
      float: Probability that the Random Probing Leakage Diagram contain at
              least an orbit of size at least n + 3.
  """

  comb = math.comb
  pa = proba_edge_add_copy(p, n)
  pun = proba_edge_11(p, n)
  nb_gadgets = nb_gun + nb_gcopy + nb_gadd + nb_gmult

  sum1 = 0
  for d in range(1, n - 1):
    sum1 += (cache[n - d]) * (
            nb_gun * (pun ** (d + 1))
            + nb_gadd * (pa ** (d + 1))
            + nb_gmult * (max_proba_mult(n, d + 1, p, cache))
            + 2 * nb_gcopy * (pa ** (d + 1))
    )
  term1 = sum1 * n * 3 * p**2

  # mult
  sum2 = 0
  for k in range(n + 3, beta + 1):
    for i in range(max(math.ceil((k - n) / 3), 2), math.floor(k / 2) + 1):
      sum2 += (
              comb(k, 2 * i)
              * (12 * p) ** (2 * i)
              * 2 ** (k - 2 * i)
              * max_proba_mult_cache[k - 2 * i]
      )  
  term2 = sum2 * nb_gmult

  # addcopy
  sum3 = 0
  for k in range(n + 3, beta + 1):
    for i in range(max(math.ceil((k - n) / 3), 2), math.floor(k / 2) + 1):
      subsum3 = 0
      for L in range(max(2 * i, k - n), k + 1):
        subsum3 += (
                   (k - L + 1)
                   * comb(L - 2, 2 * i - 2)
                   * (2 * pa) ** (k - L)
                   * (2) ** (L - 2 * i)
                   * max_proba_mult_cache[L - 2 * i]
        )
      sum3 += subsum3 * (12 * p) ** (2 * i)
  term3 = sum3 * (nb_gadd + nb_gcopy)

  # unary
  sum4 = 0
  for k in range(n + 3, beta + 1):
    for i in range(max(math.ceil((k - n) / 3), 2), math.floor(k / 2) + 1):
      subsum4 = 0
      for L in range(max(2 * i, k - n), k + 1):
        subsum4 += (
                   (k - L + 1)
                   * comb(L - 2, 2 * i - 2)
                   * (2 * pun) ** (k - L)
                   * (2) ** (L - 2 * i)
                   * max_proba_mult_cache[L - 2 * i]
        )
      sum4 += subsum4 * (12 * p) ** (2 * i - 2)

  term4 = sum4 * nb_gun * (8 * p) ** 2

  # residual term
  sum_term_last = 0
  for k in range(beta + 2):
      sum_term_last += (
            comb(beta + 1, k)
            * (12 * p) ** k
            * 2 ** (beta + 1 - k)
            * max_proba_mult_cache[beta + 1 - k]
      )
  term_last = nb_gadgets * sum_term_last

  cn3p = term1 + term2 + term3 + term4 + term_last
  return cn3p

def epsilon_circuit(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult):
  """
  We evaluate a masked arithmetic C with our compiler, composed of |nb_gun|
  unary gates, |nb_gcopy| copy gates, |nb_gadd| add gates and |nb_gmult|
  multiplications gates.
  We compute the Random probing security advantage of circuit C using our
  compiler, following Theorem 2 of the article.

  Args:
      p (float): Leakage rate.
      n (_type_): Number of shares.
      nb_gun (int): Number of unary gadgets.
      nb_gcopy (int): Number of copy gadgets.
      nb_gadd (int): Number of addition gadgets.
      nb_gmult (int): Number of multiplicattion gadgets.

  Returns:
      float: Advantage of the random probing security of the circuit C using our
      compiler. 
  """

  pa = proba_edge_add_copy(p, n)
  pun = proba_edge_11(p, n)

  # Precomputation of the probability to add k multiplication edges.
  # The cache goes from 1 to n and gives the probability to add at least k
  # multiplication edges (or addition edges, unary edges, depending of the worst
  # case).

  # beta = 50
  beta = 40 
  dps = 200

  cache = pr_add_s_mul_edges_cache(n, p, dps)
  max_proba_mult_cache = []

  for delta in range(max(n, beta + 1) + 1):
        elt1 = max_proba_mult(n, delta, p, cache)
        elt2 = pa**delta
        elt3 = pun**delta
        max_elts = max(elt1, elt2, elt3)

        max_proba_mult_cache.append(max_elts)

  cn = expr_cn(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, cache)
  cn1 = expr_cn1(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, cache)
  cn2 = expr_cn2(
        p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, cache, max_proba_mult_cache
  )
  cn3 = expr_cn3(
        p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult, beta, cache, max_proba_mult_cache
  )

  total = cn + cn1 + cn2 + cn3
  return total

################################################################################
#                                                                              #
#                             Main                                             #
#                                                                              #
################################################################################

def random_complexity(n, nb_gun, nb_gcopy, nb_gadd, nb_gmult):
  """
  Compute the random complexity of a compiled arithmetic circuit C with our compiler. 

  Args:
      n (int): Number of shares.
      nb_gun (int): Number of unary gates in C.
      nb_gcopy (int): Number of copy gates in C.
      nb_gadd (int): Number of addition gates in C.
      nb_gmult (int): Number of multiplication gates in C.

  Returns:
      int: Random complexity of the compiled circuit C.
  """
  
  nbrand_mult = n * (n - 1) / 2
  nbrand_refresh = n
  nbrefresh = nb_gun + nb_gcopy * 2 + nb_gadd + nb_gmult
  return nb_gmult * nbrand_mult + nbrefresh * nbrand_refresh


def main():
  """
  This function computes the random complexity as well as the random probing
  security advantage of AES and store it into .tsv. This is done for the
  following method : 
    - BFO23
    - JMB24
    - BNR25
    - Our work.
  """

  ps = [2**-16, 2**-12]
  colors = [("#E65100", "#FFB74D"), ("#0D47A1", "#64B5F6"), ("#FFFFFF", "#FFFFFF")]


  ## Number of gates in the AES Encryption.
  ## 11 ARK with 16 ADD
  ## 10 SBytes with 16 Sboxes with 4 mults, 21 cmults, 11 copies, 8 adds, 1 cadd
  ## 9 MixColumns with 48 adds, 32 cmults, 48 copies

  # Number of unary gadgets.
  nb_gun = 10 * 16 * (21 + 1) + 9 * 32

  # Number of copy gadgets.
  nb_gcopy = 10 * 16 * 11 + 9 * 48

  # Number of addition gadgets.
  nb_gadd = 11 * 16 + 10 * 16 * 8 + 9 * 48

  # Number of multiplication gadgets.
  nb_gmult = 10 * 16 * 4

  nb_gadgets_tot = nb_gun + nb_gcopy + nb_gadd + nb_gmult

  print("Randomness Complexity for n=12: ",
        random_complexity(12, nb_gun, nb_gcopy, nb_gadd, nb_gmult))
  print("Randomness Complexity for n=15: ",
        random_complexity(15, nb_gun, nb_gcopy, nb_gadd, nb_gmult))
  print("Randomness Complexity for n=25: ",
        random_complexity(25, nb_gun, nb_gcopy, nb_gadd, nb_gmult))
  print()

  ns = list(range(2, 21))

  for p, color in zip(ps, colors):
    logp = math.log(p, 2)
    print(f"---------------- p = 2^{int(logp)} --------------------")
    label = f"$p = 2^{{{int(logp)}}}$"

    fig, ax = plt.subplots(figsize=(14, 7))
    eps_values = [
      epsilon_circuit(p, n, nb_gun, nb_gcopy, nb_gadd, nb_gmult) for n in ns
    ]
    ns_eps = [n for n, e in zip(ns, eps_values) if e > 0]
    eps_pos = [e for e in eps_values if e > 0]

    # Our results.
    ax.plot(ns_eps, eps_pos, marker="o", linestyle="-", color=color[0],
            label=f"Our bound, {label}")

    array_out = np.column_stack((ns_eps, eps_pos))
    np.savetxt(f"p2{int(logp)}_ours.tsv", array_out, header="x\ty", 
               delimiter="\t", comments="")

    # Results of BFO23.
    eps_values_BFO23 = [expr_BFO23(p, n, nb_gadgets_tot) for n in ns]
    ax.plot(ns, eps_values_BFO23, linestyle="--", color=color[0],
            label=f"Bound from BFO23, {label}")

    array_out = np.column_stack((ns, eps_values_BFO23))
    np.savetxt(f"p2{int(logp)}_BFO23.tsv", array_out, header="x\ty", 
               delimiter="\t", comments="")

    # Results of JMB24
    eps_values_JMB24 = [expr_JMB24(p, n) for n in ns]
    ax.plot(ns, eps_values_JMB24, linestyle=":", color=color[0],
            label=f"Bound from JMB24, {label}")

    array_out = np.column_stack((ns, eps_values_JMB24))
    np.savetxt(f"p2{int(logp)}_JMB24.tsv", array_out, header="x\ty",
               delimiter="\t", comments="")

    ns_loaded = []
    eps_loaded = []
    nb_rand_loaded = []

    for n in range(len(ns) + 2):
      filename = ("prec/advantages_and_complexities_AES_BNR25_" + str(int(n))
                + "_" + str(int(math.log(p, 2))) + ".npy")

      if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True).item()
        eps = data["eps"]
        nb_rand = data["Randomness Complexities"]
        nb_rand_bit = data["Randomness Complexities with permutations"]
        ns_loaded.append(n)
        eps_loaded.append(eps)
        nb_rand_loaded.append(nb_rand)

    if ns_loaded != []:
      ax.plot(ns_loaded, eps_loaded, linestyle="None", marker="x", markersize=6,
              color=color[0], label=f" Bound from BNR25, {label}")
      array_out = np.column_stack((ns_loaded, eps_loaded))
      np.savetxt(f"p2{int(logp)}_BNR25.tsv", array_out, header="x\ty",
                 delimiter="\t", comments="")


    ax.set_xlabel("n")
    ax.set_ylabel(r"RPS Security Advantage")
    ax.set_yscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    #plt.savefig(f"AES_p{logp}.pdf", dpi=300, bbox_inches="tight")
    #plt.show()
        
    print("Complexities : ")
    for l in range (len(nb_rand_loaded)) :
          print(f"For n = {l + 2} : ")
          print(f"   BNR 25 : {nb_rand_loaded[l]}, eps = {math.log(eps_loaded[l], 2)}")
          print(f"   Ours   : {int(random_complexity(l + 2, nb_gun, nb_gcopy, nb_gadd, nb_gmult))}, eps = {math.log(eps_values[l], 2)}")
          print()
          




if __name__ == "__main__":
    main()
