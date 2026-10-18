"""Generic visual props. Metres; +Z up, +Y front. No sensing/physics model."""
import math
MATERIALS = {
    'housing': ([0.22,0.28,0.27,1],0.65,0.1),
    'dark': ([0.045,0.065,0.08,1],0.6,0.2),
    'metal': ([0.48,0.55,0.59,1],0.35,0.7),
    'white': ([0.78,0.82,0.80,1],0.55,0.05),
    'glass': ([0.025,0.18,0.24,1],0.15,0.45),
    'accent': ([0.1,0.65,0.56,1],0.4,0.1),
}
NAMES = {
 'rf_receiver':'RF antenna array', 'radar':'Radar panel',
 'eo_camera':'Visible-light camera', 'thermal_camera':'Thermal camera',
 'acoustic_array':'Microphone array', 'remote_id':'Remote ID receiver',
 'adsb_receiver':'ADS-B receiver',
}
class Shape:
    def __init__(self): self.parts=[]
    def mesh(self,name,vs,fs,mat): self.parts.append(dict(name=name,vertices=vs,faces=fs,material=mat))
    def box(self,name,center,size,mat='housing'):
        x,y,z=center; a,b,c=[v/2 for v in size]
        vs=[(x+i*a,y+j*b,z+k*c) for i,j,k in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
        self.mesh(name,vs,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],mat)
    def rod(self,name,a,b,r,mat='metal',n=20):
        d=[b[i]-a[i] for i in range(3)]; length=math.sqrt(sum(v*v for v in d)); d=[v/length for v in d]
        ref=(0,0,1) if abs(d[2])<0.9 else (0,1,0)
        u=[d[1]*ref[2]-d[2]*ref[1],d[2]*ref[0]-d[0]*ref[2],d[0]*ref[1]-d[1]*ref[0]]
        q=math.sqrt(sum(v*v for v in u));u=[v/q for v in u]
        v=[d[1]*u[2]-d[2]*u[1],d[2]*u[0]-d[0]*u[2],d[0]*u[1]-d[1]*u[0]]
        vs=[tuple(p[j]+r*(u[j]*math.cos(2*math.pi*i/n)+v[j]*math.sin(2*math.pi*i/n)) for j in range(3)) for p in (a,b) for i in range(n)]
        fs=[tuple(reversed(range(n))),tuple(range(n,2*n))]+[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
        self.mesh(name,vs,fs,mat)
    def tripod(self,height=1.55):
        for k in range(3):
            a=2*math.pi*k/3
            foot=(0.55*math.cos(a),0.55*math.sin(a),0.07)
            self.rod('tripod_leg_'+str(k),foot,(0,0,0.8),0.035,'dark')
            self.box('rubber_foot_'+str(k),foot,(0.14,0.13,0.09),'dark')
        self.rod('mast',(0,0,0.65),(0,0,height),0.04)
        self.rod('collar',(0,0,0.75),(0,0,0.88),0.065,'housing')
        self.box('electronics',(0,-0.10,0.98),(0.22,0.14,0.32))
    def lens(self,x,z,r):
        self.rod('lens_bezel',(x,0.18,z),(x,0.28,z),r,'dark',32)
        self.rod('lens_glass',(x,0.281,z),(x,0.287,z),r*0.8,'glass',32)

def build(kind):
    s=Shape();s.tripod()
    if kind=='rf_receiver':
        s.rod('array_ring',(0,0,1.58),(0,0,1.67),0.28,'housing',32)
        for k in range(6):
            a=k*math.pi/3;x,y=.22*math.cos(a),.22*math.sin(a)
            s.rod('antenna_'+str(k),(x,y,1.67),(x,y,2.06),.018,'dark')
        s.rod('centre_cap',(0,0,1.67),(0,0,1.79),.06,'white')
    elif kind=='radar':
        s.rod('swivel',(0,0,1.48),(0,0,1.67),.1,'dark')
        s.box('panel_frame',(0,0,1.95),(.83,.20,.65),'housing')
        s.box('panel_face',(0,.111,1.95),(.74,.025,.55),'white')
        for x in (-.35,.35):
            for z in (1.70,2.20):s.rod('panel_fastener',(x,.127,z),(x,.137,z),.018,'dark')
        for z in (1.77,1.87,1.97,2.07):s.box('rear_fin',(0,-.13,z),(.58,.07,.035),'dark')
    elif kind in ('eo_camera','thermal_camera'):
        s.box('yoke_base',(0,0,1.56),(.47,.20,.10),'dark')
        for x in (-.23,.23):s.box('yoke_arm',(x,0,1.72),(.06,.16,.34),'dark')
        s.box('camera_housing',(0,0,1.79),(.38,.40,.27),'white' if kind=='eo_camera' else 'housing')
        if kind=='eo_camera':
            s.lens(0,1.79,.105)
            s.box('sun_hood',(0,.13,1.95),(.43,.49,.035),'white')
        else:
            s.lens(-.085,1.79,.075);s.lens(.10,1.79,.045)
            s.box('thermal_label',(0,0.206,1.90),(.12,.008,.02),'accent')
    elif kind=='acoustic_array':
        s.rod('array_mount',(0,-.06,1.50),(0,-.06,1.85),.035,'metal')
        for k in range(6):
            a=k*math.pi/3;x,z=.35*math.cos(a),1.85+.35*math.sin(a)
            s.rod('array_spoke_'+str(k),(0,0,1.85),(x,0,z),.018)
            s.rod('microphone_'+str(k),(x,-.045,z),(x,.075,z),.05,'dark')
        s.box('array_hub',(0,0,1.85),(.14,.15,.14))
    elif kind=='remote_id':
        s.box('receiver_box',(0,0,1.65),(.29,.19,.26),'white')
        for x in (-.11,.11):s.rod('whip',(x,0,1.78),(x,0,2.02),.012,'dark')
        s.box('status_strip',(0,.101,1.65),(.13,.008,.025),'accent')
    elif kind=='adsb_receiver':
        s.box('receiver_box',(0,0,1.63),(.20,.19,.20),'housing')
        s.rod('radome',(0,0,1.73),(0,0,2.36),.025,'white')
        for k in range(4):
            a=k*math.pi/2;s.rod('radial_'+str(k),(0,0,1.75),(.24*math.cos(a),.24*math.sin(a),1.61),.009)
    else: raise ValueError(kind)
    return s.parts
