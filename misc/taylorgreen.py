import sympy as sp
from sympy.vector import CoordSys3D, gradient, laplacian
from sympy.printing import ccode

_repl_dict = {"R.x": "x[0]", "R.y": "x[1]", "R.z": "x[2]"}

def make_ccode(spexpr):
    code = ccode(spexpr)
    for key, val in _repl_dict.items():
        code = code.replace(key, val)
    lines = []
    for line in code.split("\n"):
        if line[:2] != "//":
            lines.append(line)
    code = "\n".join(lines)
    return code

if __name__ == "__main__":
    # x, y = sp.symbols("x y")
    R = CoordSys3D("R")
    t = sp.symbols("t")
    mu, rho, sigma, M, epsilon, beta, phi0, u0 = sp.symbols("mu rho sigma M epsilon beta phi0 u0", positive=True)
    beta_min = sp.sqrt(256*epsilon**4 - 184*epsilon**2 + 79)/4 
    alpha = M * sigma / epsilon * beta #_min # beta
    U = u0 * sp.exp(-2*mu/rho * t) # update
    Phi = phi0 * sp.exp(-alpha*t) # sp.symbols("Phi") #sp.exp(-t)

    u = (U * ( sp.cos(R.x) * sp.sin(R.y) * R.i - sp.sin(R.x) * sp.cos(R.y) * R.j )).doit()
    p_u = -1/4 * rho * U**2 * (sp.cos(2*R.x) + sp.cos(2*R.y)) 

    phi = Phi * sp.cos(R.x) * sp.cos(R.y)
    eta = (sigma * ( (- phi + phi**3)/epsilon - epsilon * laplacian(phi))).doit()

    p_phi = sigma * (1 - 2*epsilon**2)/(2*epsilon) * phi**2 - 3*sigma/(4*epsilon) * phi**4
    p = p_u + p_phi

    ugradu = u.dot(gradient(u.dot(R.i)))*R.i + u.dot(gradient(u.dot(R.j)))*R.j + u.dot(gradient(u.dot(R.k)))*R.k
    gradp_u = gradient(p_u)
    gradp_phi = gradient(p_phi)
    gradp = gradient(p)

    dtu = u.diff(t)
    laplu = laplacian(u)

    phigradeta = phi * gradient(eta)

    dtphi = phi.diff(t)
    ugradphi = u.dot(gradient(phi))
    lapleta = laplacian(eta)

    res_visc = rho*dtu - mu*laplu
    res_adv = rho * ugradu + gradp_u #
    res_phi = gradp_phi + phigradeta

    res_NS = rho * (dtu + ugradu) - mu*laplu + gradp + phigradeta
    q = dtphi + ugradphi - M * lapleta

    q_ccode = make_ccode(q.simplify())
    u0_ccode = make_ccode(u.dot(R.i).simplify())
    u1_ccode = make_ccode(u.dot(R.j).simplify())
    p_ccode = make_ccode(p.simplify())
    phi_ccode = make_ccode(phi.simplify())

    print("u0 = \"{}\"".format(u0_ccode))
    print("u1 = \"{}\"".format(u1_ccode))
    print("p = \"{}\"".format(p_ccode))
    print("phi = \"{}\"".format(phi_ccode))

    print("q = \"{}\"".format(q_ccode))


    exit()

    #print(q.subs(epsilon, 0.1).simplify())

    #"""
    q_lim = sp.limit(q * sp.exp(alpha*t), t, sp.oo)
    #dq = q-q_lim*sp.exp(-alpha*t)
    #qint = sp.integrate(q, (t, 0, sp.oo))

    #print(qint)

    #print(res_visc.simplify(), res_adv.simplify(), res_phi.simplify(), res_NS.simplify())

    #print(res_PF.simplify())

    q2m = sp.integrate(sp.integrate(q**2, (R.x, 0, 2*sp.pi)).simplify(), (R.y, 0, 2*sp.pi)).simplify()
    q2mt = sp.integrate(q2m, (t, 0, sp.oo))

    q2mt_beta = q2mt.diff(beta)
    q2mt_betabeta = q2mt_beta.diff(beta).doit()
    beta_min = sp.solve(q2mt_beta, beta)[0]

    q2mt_min = q2mt.subs(beta, beta_min).simplify()
    q2mt_min_epsilon = q2mt_min.diff(epsilon)

    print(beta_min, 
          q2mt_betabeta.subs(beta, beta_min).simplify(),
          q2mt_min)
    
    epsilon_min = sp.solve(q2mt_min_epsilon, epsilon)

    print(q2mt_min_epsilon, epsilon_min)
    #"""
    
    #q2m_lim = sp.limit(q2m*sp.exp(2*alpha*t), t, sp.oo)

    #print(q2m, q2m_lim)