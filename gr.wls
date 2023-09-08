#!/usr/bin/env wolframscript

(* Define the variables *)
w = {{w11, w12, w13}, {w21, w22, w23}, {w31, w32, w33}};  (* 3x3 matrix *)
d = {{d1}, {d2}, {d3}};  (* 3x1 vector *)
cov = {{cov11, cov12, cov13}, {cov12, cov22, cov23}, {cov13, cov23, cov33}};  (* 3x3 matrix *)

p = {{p0}, {p1}}
u = {{u0}, {u1}, {u2}};  (* Assuming x is a 3x1 vector for demonstration *)
t = w . u + d;  (* 3x1 vector *)
x = {{t[[1, 1]]/t[[3, 1]]}, {t[[2, 1]]/t[[3, 1]]}};  (* 2x1 vector *)
j = {
    {1/t[[3,1]], 0, -t[[1,1]]/(t[[3,1]] * t[[3,1]] )}, 
    {0, 1/t[[3,1]], -t[[2,1]]/(t[[3,1]] * t[[3,1]] )}
};  (* 2x3 matrix *)

(* Assuming the values of the variables *)
v = j . w . cov . Transpose[w] . Transpose[j];

vinv = Inverse[v];
(* Print the result *)

(* Print the result *)
mu = p - x;

ex1 =  Exp[ -0.5 * Transpose[mu] . Inverse[v] . mu]
grad1 = D[ex1, {d1}]


mud = D[mu, {d1}]
jd = D[j, {d1}]

(*Print["mud= ", mud]*)

gradv = j . w . cov . Transpose[w] . Transpose[jd] + jd . w . cov . Transpose[w] . Transpose[j]
grad4 = (Transpose[mud]. vinv . mu - Transpose[mu]. vinv . gradv . vinv. mu + Transpose[mu]. vinv . mud ) * ex1 * -0.5


(*
    x = np.array([[0], [0]])
    u = np.array([[1], [2], [3]])
    sigma_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    d = np.array([[1], [2], [3]])
*)

wval = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
dval = {{1}, {2}, {3}};
covval = {{3, 0, 0}, {0, 2, 0}, {0, 0, 1}};
xval = {{0}, {0}};
uval = {{1}, {2}, {3}};

r = grad1 /. {w11 -> wval[[1, 1]], w12 -> wval[[1, 2]], w13 -> wval[[1, 3]], 
    w21 -> wval[[2, 1]], w22 -> wval[[2, 2]], w23 -> wval[[2, 3]], 
    w31 -> wval[[3, 1]], w32 -> wval[[3, 2]], w33 -> wval[[3, 3]], 
    d1 -> dval[[1, 1]], d2 -> dval[[2, 1]], d3 -> dval[[3, 1]], 
    cov11 -> covval[[1, 1]], cov12 -> covval[[1, 2]], cov13 -> covval[[1, 3]], 
    cov21 -> covval[[2, 1]], cov22 -> covval[[2, 2]], cov23 -> covval[[2, 3]], 
    cov31 -> covval[[3, 1]], cov32 -> covval[[3, 2]], cov33 -> covval[[3, 3]], 
    p0 -> xval[[1, 1]], p1 -> xval[[2, 1]], 
    u0 -> uval[[1, 1]], u1 -> uval[[2, 1]], u2 -> uval[[3, 1]]};
Print["grad4_val= ", Simplify[r]]
















(*To help with the solver*)

(* Print the result *)
(*diff = (grad1 - grad4)[[1, 1]]*)
(*
diff = jd
diff = ReplaceAll[d1 + u0*w11 + u1*w12 + u2*w13 -> t0][diff]
diff = ReplaceAll[-d1 - u0*w11 - u1*w12 - u2*w13 -> -t0][diff]
diff = ReplaceAll[d2 + u0*w21 + u1*w22 + u2*w23 -> t1][diff]
diff = ReplaceAll[-d2 - u0*w21 - u1*w22 - u2*w23 -> -t1][diff]
diff = ReplaceAll[d3 + u0*w31 + u1*w32 + u2*w33 -> t2][diff]
diff = ReplaceAll[-d3 - u0*w31 - u1*w32 - u2*w33 -> -t2][diff]

diff = Simplify[diff]

(* Print the result *)
Print["Diff= ", diff] *)
(*diff = Simplify[diff]*)

(* Assuming the values of the variables *)
(* Print the result *)

(* Assuming the values of the variables *)
(*Print["Diff= ", diff]*)
(* Print the result *)

