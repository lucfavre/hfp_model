### Load packaged
import numpy as np
import CoolProp.CoolProp as CP

### Mathematical functions

# Dichotomy algorithm to find a root of a given function F located between a and b
def root_dichotomy(F, a, b, epsilon=1e-5, nit_max=1000):

    k=0
    while abs(b-a)>epsilon and k<nit_max:
        m=(a+b)/2

        if F(a)*F(m)<0:
            b=m
        elif F(m)*F(b)<0:
            a=m
        else:
            print(f"WARNING : no zero detected between {a} and {b}")
            converge=False
            return (m, F(m), k, converge)

    m=(a+b)/2
    converge=True
    return (m, F(m), k, converge)


# Approximation of Lambert's W-function (reciprocal of x->x*exp(x))
def approx_lamb(x):
    e=2.71828

    if x<1/e:
        return np.log(1+x)

    if x>e:
        return np.log(x)-np.log(np.log(x))

    else:
        a=(1-np.log(1+1/e))/(e-1/e)
        b=1-a*e
        return a*x+b

# Lambert's W-function
def lamb(x):

    if x<1e-4: # W(x) approx. equal to x if x<1e-4
        return x

    else:

        def reclamb(z):
            return z*np.exp(z) - x

        res=root_dichotomy(reclamb, 0, 1e3)[0] # We find the root by dichotomy for (0<x<1000) if x>1e-4

    return res



### Physical properties of fluids

# Function returning the physcial properties of a given fluid at pressure P_Pa, at saturation temperature if no subcooling is specified
def fluid_ppt(fluid, P_Pa, subcool=0):

    if fluid == 'FC87': # Properties given in Thorncroft at al. (IJHMT, 1998) at 1atm

        # Saturation temperature
        Tsat = 29.3

        # Saturation densities
        rhoL_sat = 1750
        rhoV_sat = 12.5

        # Latent enthalpy of vaporization
        hLV = 31.3e3

        # Surface tension
        sigma = 8.97e-3

        # Saturation viscosities
        nuL_sat = 458e-6 # Liquid kinematic viscosity
        muL_sat = nuL_sat * rhoL_sat
        muV_sat = 0 # Not available

        # Saturation heat capacities
        cpL_sat = 1.09e3
        cpV_sat = 0 # Not available

        # Thermal conductivities
        PrL_sat = 9.03 # Liquid Prandtl number
        lamL_sat = muL_sat * cpL_sat / PrL_sat
        lamV_sat = 0 # Not available

        return {'Tsat':Tsat, 'rhoL':rhoL_sat, 'rhoV':rhoV_sat, 'muL':muL_sat, 'muV':muV_sat, 'cpL':cpL_sat, 'cpV':cpV_sat, 'lamL':lamL_sat, 'lamV':lamV_sat, 'hLV':hLV, 'sigma':sigma}

    else:

        sat = True
        # Saturation temperature
        Tsat = CP.PropsSI('T', 'P', P_Pa, 'Q', 0, fluid) #K

        # Saturation densities
        rhoL_sat = CP.PropsSI('D', 'P', P_Pa, 'Q', 0, fluid) #kg/m3
        rhoV_sat = CP.PropsSI('D', 'P', P_Pa, 'Q', 1, fluid) #kg/m3

        # Saturation viscosities
        muL_sat = CP.PropsSI('VISCOSITY', 'P', P_Pa, 'Q', 0, fluid) #Pa.s
        muV_sat = CP.PropsSI('VISCOSITY', 'P', P_Pa, 'Q', 1, fluid) #Pa.s

        # Saturation heat capacities
        cpL_sat = CP.PropsSI('CPMASS', 'P', P_Pa, 'Q', 0, fluid) #J/kg/K
        cpV_sat = CP.PropsSI('CPMASS', 'P', P_Pa, 'Q', 1, fluid) #J/kg/K

        # Saturation heat conductivities
        lamL_sat = CP.PropsSI('L', 'P', P_Pa, 'Q', 0, fluid) #W/m/K
        lamV_sat = CP.PropsSI('L', 'P', P_Pa, 'Q', 1, fluid) #W/m/K

        # Latent enthalpy of vaporization (J/kg)
        hLV = CP.PropsSI('HMASS', 'P', P_Pa, 'Q', 1, fluid) - CP.PropsSI('HMASS', 'P', P_Pa, 'Q', 0, fluid)

        # Surface tension in J/m2
        sigma = CP.PropsSI('SURFACE_TENSION', 'P', P_Pa, 'Q', 0, fluid)

        if subcool > 0: # If a positive subcooling is specified, program will return liquid properties at TLiq = Tsat - subcool
            DTL = subcool
            sat = False

            TLiq = Tsat-DTL

            rhoL_b = CP.PropsSI('D', 'P', P_Pa, 'T', TLiq, fluid)

            muL_b = CP.PropsSI('VISCOSITY', 'P', P_Pa, 'T', TLiq, fluid)

            cpL_b = CP.PropsSI('CPMASS', 'P', P_Pa, 'T', TLiq, fluid)

            lamL_b = CP.PropsSI('L', 'P', P_Pa, 'T', TLiq, fluid)

        if not(sat) :
            return {'Tsat':Tsat, 'rhoL':rhoL_b, 'rhoV':rhoV_sat, 'muL':muL_b, 'muV':muV_sat, 'cpL':cpL_b, 'cpV':cpV_sat, 'lamL':lamL_b, 'lamV':lamV_sat, 'hLV':hLV, 'sigma':sigma}

        else :
            return {'Tsat':Tsat, 'rhoL':rhoL_sat, 'rhoV':rhoV_sat, 'muL':muL_sat, 'muV':muV_sat, 'cpL':cpL_sat, 'cpV':cpV_sat, 'lamL':lamL_sat, 'lamV':lamV_sat, 'hLV':hLV, 'sigma':sigma}


## Kurul & Podowski traditional model (International Heat Transfer Conference Digital Library, 1990)

def kp_hfp(DTw, DTL, P_bar, GL, Dh, theta, fluid, dtheta=1, Kg=1.0, Rc=0, hfc_corr=1, all_dat=False):

    ## Initialization

    # Physical parameters
    P_Pa = P_bar * 1e5 #Pressure in Pa for CoolProp physical properties
    theta_rad = theta * np.pi / 180 # Contact angle in radians
    dtheta_rad = dtheta * np.pi / 180 # Bubble tilt in radians

    # Physical constants
    g = 9.81 # Gravity acceleration

    # Saturated fluid properties
    fp_sat = fluid_ppt(fluid, P_Pa)

    [Tsat, rhoL_sat, rhoV_sat, muL_sat, muV_sat, cpL_sat, cpV_sat, lamL_sat, lamV_sat, hLV, sigma] = [fp_sat['Tsat'], fp_sat['rhoL'],
    fp_sat['rhoV'], fp_sat['muL'], fp_sat['muV'], fp_sat['cpL'], fp_sat['cpV'], fp_sat['lamL'], fp_sat['lamV'], fp_sat['hLV'], fp_sat['sigma']]

    #Subcooled fluid properties
    fp_sub = fluid_ppt(fluid, P_Pa, subcool = DTL)

    [rhoL_b, muL_b, cpL_b, lamL_b] = [fp_sub['rhoL'], fp_sat['muL'], fp_sat['cpL'], fp_sat['lamL']]

    # Sapphire substrate properties
    lam_w = 40
    cp_w = 750
    rho_w = 3975
    eta_w = lam_w/(rho_w*cp_w)

    # Wall temperature
    Tw = Tsat + DTw
    # Liquid bulk temperature
    TL = Tsat - DTL

    # Bulk liquid thermophysical properties
    (nuL_b, etaL_b) = (muL_b / rhoL_b, lamL_b / (rhoL_b * cpL_b))
    PrL_b = nuL_b / etaL_b
    UL_b = GL / rhoL_b

    # Saturation liquid thermophysical properties
    (nuL_sat, etaL_sat) = (muL_sat / rhoL_sat, lamL_sat / (rhoL_sat * cpL_sat))
    PrL_sat = nuL_sat / etaL_sat

    # Vapor thermophysical properties
    (nuV_sat, etaV_sat) = (muV_sat / rhoV_sat, lamV_sat / (rhoV_sat * cpV_sat))
    PrV = nuV_sat / etaV_sat

    #Liquid wall friction velocity
    Re_Dh = GL*Dh/muL_b # Hydraulic diameter Reynolds number
    Cf = 0.036 * ( Re_Dh**(-0.1818) ) # MacAdams friction factor

    ### SINGLE-PHASE HEAT TRANFER COEFFICIENT
    Nu_L =  ( (Cf / 2) * (Re_Dh - 1000) * PrL_b) / (1 + 12.7 * np.sqrt(Cf / 2) * (PrL_b**(2/3) - 1)) # Nusselt number using Gnielinski correlation
    h_cL = lamL_b * Nu_L / Dh # Resulting liquid HTC
    h_cL = h_cL * hfc_corr #  Corrective factor (e.g. equal to 4.0 if only one side of the channel is heated)

    ### INITIALIZING FLUXES & AREAS
    (phiw_calc, phi_cL, phi_e, phi_q)=(0,0,0,0)
    (AcL, Aq)=(1,0)
    (rlo, nsit, tw, f, Aq)=(0,0,0,0,0)

    ### BOILING REGIMES

    # Boiling criterion
    if DTw>0:
        boil=True  # We consider boiling only  if Twall > Tsat
    else:
        boil=False # If Twall<=Tsat we only consider liquid convective heat flux

    if not(boil) :
        # Areas
        AcL = 1. # We only have wall area impacted by liquid

        # Liquid convective flux
        phi_cL = AcL * h_cL * (Tw - TL)

        # Total flux
        phiw_calc = phi_cL # Pure liquid convection

    else: #We enter in the boiling model

        # Lemmert - Chawla correlation
        nsit = (210 * DTw)**1.8
        # Unal correlation
        a = DTw * lam_w / (2 * rhoV_sat * hLV * np.sqrt(np.pi * eta_w))
        b = DTL / (2 * (1 - rhoV_sat/rhoL_sat))
        U0 = 0.61
        phi = max(1, (UL_b/U0)**0.47)
        dlo = 2.42e-5 * ((P_bar * 1e5)**0.709) * a / np.sqrt(b * phi)

        rlo = dlo/2
        # Cole frequency
        f = np.sqrt((4/3) * g * abs(rhoV_sat - rhoL_sat)/(rhoL_sat * dlo))

        tw = 1/f

        Fa = 4
        Aq = min(1, Fa * np.pi * (rlo**2) * nsit)

        ## SINGLE-PHASE FORCED CONVECTION HEAT FLUX
        # Liquid-facing wall area
        AcL = 1 - Aq
        # Resulting single-phase heat flux
        phi_cL = h_cL * (Tw - TL) * AcL

        # Boiling heat flux
        phi_e = (1/6) * np.pi * (dlo**3) * rhoV_sat * hLV * f * nsit

        # Quenching heat flux
        phi_q = Aq * 2 * lamL_b * (Tw - TL) / np.sqrt(np.pi * etaL_sat * tw)

        # Total heat flux
        phiw_calc = phi_cL + phi_q + phi_e

    #RETURNING VARIABLES
    fluxes={'phi_w':phiw_calc, 'phi_cL':phi_cL, 'phi_e':phi_e, 'phi_q':phi_q}

    areas={'AcL':AcL, 'Aq':Aq}

    if all_dat==True:
        extra_var={'nsit':nsit, 'f':f, 'tw':tw, 'Rlo':rlo}

    else:
        extra_var=[]

    return [fluxes, areas, extra_var]

### PHYSICS FUNCTION

# Reichardt dimensionless liquid velocity profile
def uplus_reich(yplus):
    kappa = 0.41
    chi = 11
    c = 7.4
    uplus = (1 / kappa) * np.log(1 + kappa * yplus) + c * (1 - np.exp(-yplus / chi) + (yplus / chi) * np.exp(- yplus / 3)  ) #Reichardt law

    return uplus

# Reichardt dimensionless wall shear rate profile
def duplus_reich(yplus):
    kappa = 0.41
    chi = 11
    c = 7.4
    duplus=(1 / (1 + kappa * yplus)) + (c / chi) * (np.exp(-yplus / chi) + (1 - yplus / 3) * np.exp(-yplus / 3) )

    return duplus

# Shi et al. (JFM, 2021) drag coefficient calculation
def CD_shi_adim(Re, Sr, Lr, Lu):
    # Re = Db * abs(Urel) / nuL ;
    # Sr = gamma * Db / Urel ;
    # Lr = 2 * L / Db ;
    # Lu = L * abs(Urel) / nuL

    #Uniform flow drag coefficient
    def drag_unif(r):
        #Unif. drag. from Mei & Klaunser validated by Scheiff et al. (2020)
        res = (16 / r) * (1 + (8 / r + (1/2) * ( 1 + 3.315 / (r**(1/2)) ) )**(-1)  )
        return res

    CD_unif=drag_unif(Re)

    #Drag correction

    #Low Re correction
    fpD = 1 / (1 + 0.16 * Lu * (Lu + 4))
    b = 1 + np.tanh(0.012 * (Re**0.8)) + np.tanh(0.07 * (Re**0.8))
    delta_CD_Win = ( ((3/8) * (Lr**-1) + (3/64) * (Lr**-4)) / (1 - (3/8) * (Lr**-1) - (3/64) * (Lr**-4)) ) - (1/16) * ((Lr**-2) + (3/8) * (Lr**-3)) * Sr
    delta_CD_W_lowRe = fpD * (b**2) * delta_CD_Win

    #High Re correction
    delta_CDu_W = 0.47 * (Lr**-4) + 5.5e-3 * (Lr**-6) * (Re**(3/4))
    delta_CDom_U = 2e-3 * (abs(Sr)**1.9) * Re
    delta_CDom_WU = 0.05 * (Lr**(-7/2)) * Sr * (Re**(1/3))

    delta_CD_W_highRe = delta_CDu_W + delta_CDom_U + delta_CDom_WU

    #Total correction
    delta_CD_W = delta_CD_W_lowRe + (1 - np.exp(-0.07 * Re)) * delta_CD_W_highRe

    #Resulting total drag coefficient
    CD_W = CD_unif * (1 + delta_CD_W)
    return CD_W

# Shi et al. (JFM, 2021) lift coefficient calculation
def CL_shi_adim(Re, Sr, Lr, Lu, Lom):
    # Re = Db * abs(Urel) / nuL ;
    # Sr = gamma * Db / Urel ;
    # Lr = 2 * L / Db ;
    # Lu = L * abs(Urel) / nuL
    # Lom=L*sqrt(gamma/nuL)

    # Dimensionless parameter epsilon = Stokes length to Saffman length ratio
    eps=(abs(Sr) / Re)**(1/2)

    # Lift coefficient

    # Viscous lift terms
    fL = np.exp(-0.22 * (eps**0.8) * (Lom**2.5))
    fpL = 1 / (1 + 0.13 * Lu * (Lu + 0.53))
    b = 1+ np.tanh( 0.012 * (Re**0.8)) + np.tanh(0.07 * (Re**0.8))
    g = -2 * np.tanh(0.01 * Re)
    C_Lu_Win = 0.5 * (1 + (1/8) * (Lr**(-1)) - (33/64) * (Lr**(-2)))
    cT1 = 1 - np.exp(-0.22 * (Re**0.6))
    C_Lu_W_Reinf = -(3/8) * (Lr**(-4)) * (1 + (1/8) * (Lr**(-3)) + (1/6) * (Lr**(-5)))
    cT2 = 15 * np.tanh(0.01 * Re)

    # Resulting viscous lift coefficient
    C_Lu_W = fL * fpL * (b**2) * ((Lr/3)**g) * C_Lu_Win + cT1 * (C_Lu_W_Reinf + cT2 * (Re**-1) * (Lr**-4))

    # Shear lift terms
    J_Linf = 2.254
    J_Leps = J_Linf * (1 + 0.2 * (eps**0.2))**(-3/2)
    hpL = 1 - np.exp(-(11/96) * (np.pi**2) * (Lom / J_Leps) * (1 + (9/8) * (Lr**-1) - (1271/3520) * (Lr**-2)))
    C_Lom_U_lowRe = (8 / (np.pi**2)) * (Sr / abs(Sr)) * eps * J_Leps
    aG = 0.23
    bG = 13
    I_ws = aG * (Lr**(-7/2)) * (1 + bG * (Re**(-1/2)))
    C_Lom_U_highRe = (2/3) * Sr * (1 - 0.07 * abs(Sr)) * (1 + 16 * (Re**(-1))) / (1 + 29 * (Re**(-1)))
    cT3=(1 - np.exp(-0.3 * Re))

    # Resulting shear lift coefficient
    C_Lom_W = hpL * C_Lom_U_lowRe + cT3 * (1 + I_ws) * C_Lom_U_highRe

    # Total lift coefficient
    C_L_tot = C_Lu_W + C_Lom_W

    return C_L_tot


## BUBBLE DEPARTURE RADIUS CALCULATION
def departure_radius(Ja, utau, theta, dtheta, sigma, rhoL, rhoV, nuL, PrL, Kg=2):
    #Ja is the wall superheat Jakob number, theta and dtheta must be given in radians.

    #Gravity constant
    g = 9.81

    #Bubble foot to radius ratio
    rw_s_r = np.sin(theta)

    #Capillary term hindering the departure
    fCx = 2.5 * rw_s_r * dtheta * (np.sin(theta)) * (np.cos(dtheta)) / ( (np.pi/2)**2 - dtheta**2)

    #Added mass coefficient promoting departure (see Favre et al. (IJHMT, 2023))
    CAMx = 0.636

    #Liquid viscosity
    muL = nuL * rhoL

    if utau == 0: #If no liquid velocity --> Only buoyancy triggers departure
        rsol = np.sqrt( (3/4) * sigma * fCx / ( (rhoL - rhoV) * g ) )
        return [rsol, True]

    K = Kg #Growth constant

    def bdf_adim(rguess): #Dimensionless force balance

        dguess = 2 * rguess
        yplus = rguess * utau / nuL

        #Reichardt law for dimensionless wall liquid velocity profile
        uplus = uplus_reich(yplus)
        duplus = duplus_reich(yplus)

        #Liquid velocity and shear rate at y+
        ULiq = uplus * utau
        gamma=( (utau**2) / nuL ) * duplus

        #Dimensionless numbers
        Reb = 2 * rguess * ULiq / nuL #Before detachment, Ub=0 -> Urel = UL- Ub = UL
        Sr = gamma * dguess / ULiq
        Fr = rhoL * (ULiq**2) / ((rhoL - rhoV) * g * rguess)
        Ca = muL * ULiq / sigma

        #Dimensionless distances for Shi et al. drag coefficent calculation
        Lr = 2 * rguess / dguess
        Lu = rguess * abs(ULiq) / nuL
        CD = CD_shi_adim(Reb, Sr, Lr, Lu)

        #Dimensionless force balance computation
        res = CAMx * (K**2) * (Ja**2) / PrL + (1/3) * Reb / Fr + (1/8) * CD * Reb - (1/2) * fCx / Ca

        return res

    #We solve the non-dimensional force balance with a dichotomy
    solve = root_dichotomy(bdf_adim, 1e-12, 1e-1, epsilon=1e-10)

    rsol=solve[0]
    conv=solve[-1]

    return [rsol, conv]


## BUBBLE SLIDING ACCELERATION CALCULATION
def bub_accel(ub, r, rdot, utau, nuL, rhoL, rhoV, sigma, theta=0, dtheta=0):

    # Bubble diameter
    d = 2 * r

    # Gravity constant
    g = 9.81

    # Bubble foot to radius ratio
    rw_s_r = np.sin(theta)

    # Capillary term hindering the departure
    fCx = 2.5 * rw_s_r * dtheta * (np.sin(theta)) * (np.cos(dtheta)) / ( (np.pi/2)**2 - dtheta**2)

    # Added Mass coefficient
    CAMx = 0.636

    # Reichardt law for dimensionless velocity and shear rate
    yplus = r * utau / nuL
    uplus = uplus_reich(yplus)
    duplus = duplus_reich(yplus)

    # Resulting in liquid velocity and shear at the wall
    ULiq = uplus * utau
    gamma =( (utau**2) / nuL ) * duplus
    # Relative velocity
    Urel = ULiq - ub

    #Non-dimensional numbers
    Reb = d * abs(Urel) / nuL

    if Urel == 0:
        Sr = 0 # No shear seen by the bubble if zero relative velocity
    else:
        Sr = gamma * d / Urel

    #Dimensionless distances for drag coefficent calculation
    Lr = 2 * r / d
    Lu = r * abs(Urel) / nuL

    if Reb == 0:
        CD=0 #No drag force if zero relative velocity
    else:
        CD = CD_shi_adim(Reb, Sr, Lr, Lu) #Otherwise, Shi drag coefficient

    # Bubble acceleration calculation (Favre et al., IJHMT, 2022)
    res = (3/8) * (1/r) * CD * (rhoL / rhoV) * Urel * abs(Urel) + (rhoL / rhoV - 1) * g + 3 * (rdot / r) * ( (rhoL / rhoV) * CAMx * Urel - ub)  - (3/4) * sigma * fCx / (rhoV * (r**2))

    res /= (1 + (rhoL/rhoV) * CAMx)

    return res


## DIMENSIONLESS FORCE BALANCE PERPENDICULAR TO THE WALL
def bdf_orth(r, ub, Ja, utau, rhoL, rhoV, nuL, PrL, Kg=2):

    # Bubble diameter
    d = 2 * r

    # Bubble growth rate
    K = Kg

    # Added mass coefficients (Favre et al., IJHMT 2022)
    CAM_y1 = 0.27
    CAM_y2 = 0.326
    CAM_y3 = 8.77e-3

    # Reichardt law for dimensionless velocity and shear rate
    yplus = r * utau / nuL
    uplus = uplus_reich(yplus)
    duplus = duplus_reich(yplus)

    # Resulting in liquid velocity and shear at the wall
    ULiq = uplus * utau
    gamma =( (utau**2) / nuL ) * duplus
    # Relative velocity
    Urel = ULiq - ub

    # Dimensionless numbers
    Reb = d * abs(Urel) / nuL
    Sr = gamma * d / Urel
    rho_st = rhoL / rhoV

    #Dimensionless distances for drag coefficent calculation
    Lr = 2 * r / d
    Lu= r * abs(Urel) / nuL
    Lom=r * np.sqrt(gamma / nuL)

    #Lift coefficient
    CL = CL_shi_adim(Reb, Sr, Lr, Lu, Lom)

    # Force balance normal to the wall
    res = rho_st * (CL / 8 + CAM_y3 / 3) - (1/3) * (rho_st * (2 * CAM_y1 + CAM_y2) + (2/3)) * (K**2 * Ja**2 / (PrL * Reb))**2

    return res


#Function computing the dimensionless area covered by a sliding bubble Aq_st = Aq / (pi * Rlo**2)
def aq_sl_1b(lsl_st, Rd_s_Rlo):

    if  lsl_st <= 1-Rd_s_Rlo :
        Aq_st=1

    elif lsl_st > 1 + Rd_s_Rlo:
        Aq_st = 0.5*(1+(Rd_s_Rlo)**2) + (lsl_st/np.pi)*(1+Rd_s_Rlo)

    else:
        xa = 1 - Rd_s_Rlo
        xb = 1 + Rd_s_Rlo
        ya = 1
        yb = 0.5*(1+(Rd_s_Rlo)**2) + (lsl_st/np.pi)*(1+Rd_s_Rlo)


        Aq_st = ya + (lsl_st -xa)/(xb - xa) * (yb - ya)

    return Aq_st



def new_hfp(DTw, DTL, P_bar, GL, Dh, theta, fluid, dtheta=1, Kg=1.0, Rc=0, hfc_corr=1, all_dat=False, chf_triplet=False, corr_nsit=False):
    # Units : ([K], [K], [bar], [kg/m2/s], [m], [deg], [-], [deg], [-], [m], [-])

    ### INITIALIZATION

    # Physical parameters
    P_Pa = P_bar * 1e5 #Pressure in Pa for CoolProp physical properties
    theta_rad = theta * np.pi / 180 # Contact angle in radians
    dtheta_rad = dtheta * np.pi / 180 # Bubble tilt in radians

    # Physical constants
    g = 9.81 # Gravity acceleration

    # Saturated fluid properties
    fp_sat = fluid_ppt(fluid, P_Pa)

    [Tsat, rhoL_sat, rhoV_sat, muL_sat, muV_sat, cpL_sat, cpV_sat, lamL_sat, lamV_sat, hLV, sigma] = [fp_sat['Tsat'], fp_sat['rhoL'],
    fp_sat['rhoV'], fp_sat['muL'], fp_sat['muV'], fp_sat['cpL'], fp_sat['cpV'], fp_sat['lamL'], fp_sat['lamV'], fp_sat['hLV'], fp_sat['sigma']]

    #Subcooled fluid properties
    fp_sub = fluid_ppt(fluid, P_Pa, subcool = DTL)

    [rhoL_b, muL_b, cpL_b, lamL_b] = [fp_sub['rhoL'], fp_sat['muL'], fp_sat['cpL'], fp_sat['lamL']]


    # Wall temperature
    Tw = Tsat + DTw
    # Liquid bulk temperature
    TL = Tsat - DTL

    # Bulk liquid thermophysical properties
    (nuL_b, etaL_b) = (muL_b / rhoL_b, lamL_b / (rhoL_b * cpL_b))
    PrL_b = nuL_b / etaL_b
    UL_b = GL / rhoL_b

    # Saturation liquid thermophysical properties
    (nuL_sat, etaL_sat) = (muL_sat / rhoL_sat, lamL_sat / (rhoL_sat * cpL_sat))
    PrL_sat = nuL_sat / etaL_sat

    # Vapor thermophysical properties
    (nuV_sat, etaV_sat) = (muV_sat / rhoV_sat, lamV_sat / (rhoV_sat * cpV_sat))
    PrV = nuV_sat / etaV_sat

    #Liquid wall friction velocity
    Re_Dh = GL*Dh/muL_b # Hydraulic diameter Reynolds number
    Cf = 0.036 * ( Re_Dh**(-0.1818) ) # MacAdams friction factor
    tauw = (Cf / 2) * rhoL_b * (GL / rhoL_b)**2 # MacAdams Correlation
    u_tau=np.sqrt(tauw / rhoL_b)

    # Dimensionless numbers of the boiling flow
    Jaw = DTw * rhoL_sat * cpL_sat / (hLV * rhoV_sat) # Superheated wall Jakob number
    JaL = DTL * rhoL_b * cpL_b / (hLV * rhoV_sat) # Subcooled liquid Jakob number
    Jaw_red = DTw * cpL_sat / hLV # Superheated wall reduced Jakob number
    JaL_red = DTL * cpL_sat / hLV # Subcooled liquid reduced Jakob number
    rho_st = rhoL_sat / rhoV_sat # Phase densities ratio

    Lc = np.sqrt(sigma / (g * (rhoL_sat - rhoV_sat))) # Capillary length
    Re_tau = u_tau * Lc / nuL_sat # Local wall Reynolds number

    ### SINGLE-PHASE HEAT TRANFER COEFFICIENT
    Nu_L =  ( (Cf / 2) * (Re_Dh - 1000) * PrL_b) / (1 + 12.7 * np.sqrt(Cf / 2) * (PrL_b**(2/3) - 1)) # Nusselt number using Gnielinski correlation
    h_cL = lamL_b * Nu_L / Dh # Resulting liquid HTC
    h_cL = h_cL * hfc_corr #  Corrective factor (e.g. equal to 4.0 if only one side of the channel is heated)

    ### INITIALIZING FLUXES & AREAS
    (phiw_calc, phi_cL, phi_e_coal_st, phi_e_coal_sl, phi_e_sl, phi_e_tot, phi_q, phi_cV)=(0,0,0,0,0,0,0,0)
    (AcL, Aq_tot, AcV)=(1,0,0)
    (Rlo, R_avg,tg_dep, tg_lo, tg_avg, tw, t_star, tq, f, Aq_1b_avg, Abub_tot)=(0,0,0,0,0,0,0,0,0,0,0)


    ### MODEL OPTIONS
    sliding = True
    [lsl, lsl_0, Rd, Rlo, Rsl, s, s_a, s_b]=[0,0,0,0,0,0,0,0]
    coalesce = False
    [nsit_0, nsit_a, nsit_coal_st, nsit_sl, nsit_coal_sl]=[0,0,0,0,0]

    ### BOILING REGIMES

    # Boiling criterion
    if DTw>0:
        boil=True  # We consider boiling only  if Twall > Tsat
    else:
        boil=False # If Twall<=Tsat we only consider liquid convective heat flux

    if not(boil) :
        # Areas
        AcL = 1. # We only have wall area impacted by liquid

        # Liquid convective flux
        phi_cL = AcL * h_cL * (Tw - TL)

        # Total flux
        phiw_calc = phi_cL # Pure liquid convection

    else: #We enter in the boiling model

	    ## NUCLEATION SITE DENSITY

        pp_Mpa = P_bar / 10 # Pressure in MPa, as required in Li et al. (NED, 2018) correlation

        # Intermediate functions needed in Li et al. correlation
        fP = 26.006 - 3.678 * np.exp(-2 * pp_Mpa) - 21.907 * np.exp(-pp_Mpa / 24.065)
        A = -0.0002 * (pp_Mpa**2) + 0.0108 * pp_Mpa + 0.0119
        B = 0.122 * pp_Mpa + 1.988
        fsup = DTw**(A * DTw + B)

        theta_0=theta_rad
        Tc = 647.096 #K
        T0 = 25 + 273.15 # Reference temperature of 25 degC
        gamma = 0.719
        ftheta = (1 - np.cos(theta_0)) * (((Tc - Tsat)/(Tc - T0))**gamma) # Contact angle function depending on temperature from Li et al.

        N0 = 1000 # sites / m2

        # NSD formulation by Li et al.
        nsit_0 = N0 * ftheta * np.exp(fP) * fsup
        if corr_nsit:
           # nsit_corr = (DTw**(2 - 0.3 * DTw**0.5)) # NSD correction if needed G2000
            nsit_corr = (DTw**(2 - 0.3 * DTw**0.5)) # NSD correction if needed G500
        else:
            nsit_corr = 1
        nsit_0 *= nsit_corr


        ## TIME SCALES

        t_star = (lamL_sat / h_cL)**2 / (np.pi * etaL_sat) # Theoretical time when forced convection = transient conduction

        # wait time by Mikic & Rohsenow formulation, modified by Yeoh et al.
        C1 = (1 + np.cos(theta_rad)) / np.sin(theta_rad)
        C2 = 1 / np.sin(theta_rad)

        if Rc>0:
            rc=Rc

        else: # If no cavity radius is specified, we compute it using Han & Griffith formulation
            rc = 2 * sigma * Tsat / (rhoV_sat * hLV * DTw)

        tw= 1 / (np.pi * etaL_sat) * ( (DTw + DTL) * rc * C1 / (DTw - 2 * sigma * Tsat / (C2 * rhoV_sat * hLV * rc))  )**2 # Wait time formulation


        ## BUBBLE DEPARTURE, SLIDING AND LIFT-OFF DIAMETER

        K=Kg # Specified growth constant Kg

        # Reduced correlation for lift-off diameter
        Dlo_max = Lc * 4582.5 * (PrL_sat**(-0.005)) * (rhoL_sat / rhoV_sat)**(-0.36) * (Jaw_red**1.15) * (1 + JaL_red)**(-6.68) * (1 + Re_tau)**(-0.53)
        Rlo_max = Dlo_max / 2


        # We compute the departure radius using the force balance from Favre et al. (IJHMT, 2023)
        Rd = departure_radius(Jaw, u_tau, theta_rad, dtheta_rad, sigma, rhoL_sat, rhoV_sat, nuL_sat, PrL_sat, Kg=K)[0]

        # This yields the departure time tg_dep
        tg_dep = ( (Rd / (K * Jaw) )**2 ) / etaL_sat

        # Then yielding the bubble nucleation frequency
        f = 1 / (tg_dep + tw)

        # Nucleation site geometrical suppression
        Asit = tg_dep * f * np.pi * (Rd**2)
        nsit_a = lamb(nsit_0 * Asit) / Asit #Actual active nucleation site density
        nb_sites = tg_dep * f * nsit_a # Actual number of bubbles on their site in average

        s_b = 1 / (2 * np.sqrt(nb_sites)) # Average distance between two bubbles on their site
        s_a = 1 / (2 * np.sqrt(nsit_a)) # Average distance between two nucleation sites

        # Initialization of bubble parameter for sliding simulation
        (R, t, ub, lsl, BdF_y) = (Rd, tg_dep, 1e-9, 0, -1.0e2)
        dt = 1e-12 # Sliding simulation time step
        Rdot = 0.5 * K * Jaw * np.sqrt(etaL_sat/t) # Bubble growth rate

        # Bubble slides until :
        # - it reaches its max lift-off diam. (R < Rlo_max)
        # - OR it reaches an other bubble on its site (lsl = s_b)
        # - OR force balance is positive perpendicular to the wall
        while (lsl < 1.0 * s_b) and (R < Rlo_max): # and and (BdF_y < 0):

            # Computing bubble acceleration
            ub_dot = bub_accel(ub, R, Rdot, u_tau, nuL_sat, rhoL_sat, rhoV_sat, sigma)
            # Computing velocity increment
            dub = ub_dot * dt

            if abs(dub) / abs(ub) > 0.03:
            # If bubble velocity increment is larger than 3% of current bubble velocity,
            # we reduce the time step to capture the steep acceleration
                dt = 0.03 * abs(ub) / abs(ub_dot)

            elif abs(dub) / abs(ub) <0.005:
            # If bubble velocity increment is smaller than 0.5% of current bubble velocity,
            # we relax the time step to accelerate the sliding computation
                dt = 0.005 * abs(ub) / abs(ub_dot)

            ub += ub_dot * dt # Update of liquid velocity
            lsl += ub * dt # Update of sliding length
            t += dt # Update of time step
            R = K * Jaw * np.sqrt(etaL_sat * t) # Update of bubble radius
            Rdot = 0.5 * K * Jaw * np.sqrt(etaL_sat / t) # Update of bubble growth rate

            #BdF_y = bdf_orth(R, ub, Jaw, u_tau, rhoL_sat, rhoV_sat, nuL_sat, PrL_sat, Kg=K)
        # End of sliding loop

        lsl_0 = lsl #Final sliding length
        tg_lo = t #Final lift-off time

        if lsl >= s_b:
            #If the bubbles reach the inter-bubble distance s_b, they lift-off due to coalescence
            lsl_0 = s_b
            Rsl = R # Single bubble sliding radius before lift-off by coalescence
            Rlo = (Rd**3 + R**3)**(1/3) #Final bubble lift-off diameter after coalescence
        else:
            #Otherwise, it lifts-off alone at Rlo
            Rsl = R
            Rlo = R

        # Single bubble sliding area
        lsl_st = lsl_0 / Rlo # Dimensionless sliding length
        Rd_s_Rlo = Rd / Rlo # Departure to lift-off radius ratio

        Aq_st = aq_sl_1b(lsl_st, Rd_s_Rlo) # Dimensionless sliding area
        Aq_1b_sl = Aq_st * np.pi * (Rlo**2) # Final sliding area for one bubble


        # Static coalescence computation
        Pcoal_st = 1 - np.exp(-nb_sites * np.pi * (2 * Rd)**2) # Probability of static coal.
        nsit_coal_st = Pcoal_st * nsit_a # Sites leading to static coal. bubbles
        nsit_sl=nsit_a * (1 - Pcoal_st) # Other sites yield sliding bubbles
        Rco_st=(2**(1/3)) * Rd # Static coalescence radius of bubbles at Rd

        if lsl >= s_b:
            # If the bubbles slide up to coalescence with a static one,
            # this means half of the sites yield sliding bubbles and
            # the other half yield static bubbles that will be absorbed at approximately R=Rd

            nsit_sl /= 2 # Actual site density yielding sliding bubbles
            nsit_coal_sl = nsit_sl # Remaning half site density yield bubbles coalesced by sliding ones
        else:
            nsit_coal_sl = 0


        ## QUENCHING HEAT FLUX

        #Quenching areas
        Ab_sl = nsit_sl * Aq_1b_sl # Total sliding area covered by bubbles
        Ab_coal_st = nsit_coal_st * np.pi * (Rd**2) # Static coalescence quenching area

        Aq_1b_avg = (1/(nsit_sl + nsit_coal_st)) * ( nsit_coal_st * Ab_coal_st + nsit_sl * Ab_sl) # Average area covered by a bubble
        tg_avg = (1/nsit_a) * (nsit_coal_st * tg_dep + nsit_sl * tg_lo + nsit_coal_sl * tg_dep) # Average growth time
        R_avg = (1/nsit_a) * ( (nsit_coal_st / 2) * Rco_st + nsit_sl * Rlo) # Average bubble radius of ejected bubbles

        tq = min(tw,t_star) # Quenching time
        # If tw > t_star, we first have transient conduction up to tq = tw -  t_star, and then forced convection from tq = tw - t_star to tw
        Ab_tot = Ab_sl + Ab_coal_st # Total wall area seeing quenching from coalescing or sliding bubbles
        Aq_tot = min(Ab_tot * tq * f, 1) # Total time averaged area under quenching

        # Resulting quenching heat flux
        phi_q = Aq_tot * (2 * lamL_b * (Tw - TL)) / np.sqrt(np.pi * etaL_b * tq)

        #Bubble-influenced area
        Abub_tot = min(Ab_tot,1)

        ## DRY VAPOR HEAT FLUX
        # Average dry area on the wall
        AcV = 0.5 * nsit_sl * np.pi * ((2/3) * Rsl)**2 * tg_lo * f + 0.5 * nsit_sl * np.pi * ((2/3) * Rd)**2 * tg_dep * f + nsit_coal_st * np.pi * ((2/3) * Rd)**2 * tg_dep *f
        # Average bubble size during its growth up to R_avg
        Rb_moy = (2/3) * R_avg
        # Equivalent conductive heat transfer coefficien
        h_cV = lamV_sat / Rb_moy
        #Resulting dry vapor heat flux
        phi_cV = AcV * h_cV * DTw

        ## SINGLE-PHASE FORCED CONVECTION HEAT FLUX
        # Liquid-facing wall area
        AcL = max(1 - Aq_tot - AcV, 0 )
        # Resulting single-phase heat flux
        phi_cL = h_cL * (Tw - TL) * AcL


        ## BOILING HEAT FLUXES
        phi_e_sl = nsit_sl * f * rhoV_sat * hLV * (4/3) * np.pi * (Rsl**3) # Sliding bubbles boiling flux
        phi_e_coal_sl = nsit_coal_sl * f * rhoV_sat * hLV * (4/3) * np.pi * (Rd**3) # Bubbles coalesced by sliding boiling flux
        phi_e_coal_st = nsit_coal_st * f * rhoV_sat * hLV * (4/3) * np.pi * (Rd**3) # Static coalescing bubbles boiling flux

        phi_e_tot = phi_e_sl +  phi_e_coal_sl + phi_e_coal_st # Total boiling flux

        phiw_calc = phi_cL + phi_cV + phi_q + phi_e_tot # Total calculated wall heat flux

    #RETURNING VARIABLES
    fluxes={'phi_w':phiw_calc, 'phi_cL':phi_cL, 'phi_e_coal_st':phi_e_coal_st, 'phi_e_coal_sl':phi_e_coal_sl, 'phi_e_sl':phi_e_sl,
            'phi_q':phi_q, 'phi_cV':phi_cV}

    areas={'AcL':AcL, 'Aq_tot':Aq_tot, 'AcV':AcV}

    if all_dat==True:
        #extra_var=[lsl, s, s_a, Rd, Rsl, nsit_a, nsit_coal_st, nsit_sl, f, tg_dep, tg_avg, tw, t_star, Aq_1b_avg, np.pi*(R_avg**2), Abub_tot, phi_cL/phiw_calc, phi_q/phiw_calc]
        #extra_var={'lsl0':lsl*1000, 's':s*1000, 'Rd':Rd*1000, 'Rco_sl':Rsl*1000, 'Rlo':Rlo*1000, 'nsit0':nsit_0, 'nsita':nsit_a, 'nsitco':nsit_coal_st, 'nsitsl':nsit_sl,  'nsitcosl':nsit_coal_sl, 'f':f, 'tgd':tg_dep, 'tgavg':tg_avg, 'tw':tw, 'tstar':t_star, 'Aq1b':Aq_1b_avg, 'piR2':np.pi*(R_avg**2), 'Abtot':Abub_tot, 'phicL\%':phi_cL/phiw_calc, 'phiq\%':phi_q/phiw_calc}
        extra_var={'nsit_a':nsit_a, 'f':f, 'tw':tw, 'tq':tq, 'tg':tg_avg, 'lsl':lsl_0, 's_b':s_b, 's_a':s_a, 'Rd':Rd, 'Rsl':Rsl}

    else:
        extra_var=[]

    if chf_triplet==True:
        extra_var.append(nsit_a*f*tg_avg*np.pi*(0.66*R_avg**2))


    return [fluxes, areas, extra_var]
