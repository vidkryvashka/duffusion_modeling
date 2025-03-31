The diffusion equation from the PDF for the dissolved phase is:

𝐽
𝐷
=
−
𝐴
Δ
𝑧
𝐷
𝑤
∂
2
(
𝐶
𝑤
𝜃
)
∂
𝑧
2
J 
D
​
 =−AΔzD 
w
​
  
∂z 
2
 
∂ 
2
 (C 
w
​
 θ)
​
 

In your case, we’ll simplify this for a 2D system with isotropic diffusion (assuming the diffusion coefficient 
𝐷
D is constant in both 
𝑥
x and 
𝑧
z directions), and since 
𝜃
θ (volumetric water content) isn’t specified, we’ll treat 
𝑞
q as the concentration 
𝐶
𝑤
C 
w
​
  directly. The 2D diffusion equation becomes:

∂
𝑞
∂
𝑡
=
𝐷
(
∂
2
𝑞
∂
𝑥
2
+
∂
2
𝑞
∂
𝑧
2
)
∂t
∂q
​
 =D( 
∂x 
2
 
∂ 
2
 q
​
 + 
∂z 
2
 
∂ 
2
 q
​
 )

The second derivatives are approximated as:
∂
2
𝑞
∂
𝑥
2
≈
𝑞
𝑖
+
1
,
𝑗
−
2
𝑞
𝑖
,
𝑗
+
𝑞
𝑖
−
1
,
𝑗
Δ
𝑥
2
∂x 
2
 
∂ 
2
 q
​
 ≈ 
Δx 
2
 
q 
i+1,j
​
 −2q 
i,j
​
 +q 
i−1,j
​
 
​
 
∂
2
𝑞
∂
𝑧
2
≈
𝑞
𝑖
,
𝑗
+
1
−
2
𝑞
𝑖
,
𝑗
+
𝑞
𝑖
,
𝑗
−
1
Δ
𝑧
2
∂z 
2
 
∂ 
2
 q
​
 ≈ 
Δz 
2
 
q 
i,j+1
​
 −2q 
i,j
​
 +q 
i,j−1
​
 
​