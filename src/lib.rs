
extern crate ndarray;

use ndarray::prelude::*;
use std::collections::{HashMap,BTreeMap};
use ndarray::stack;

#[derive(Debug)]
pub struct  FentonKarmaBR{
	pub params:HashMap<String,f64>,
    pub initial_conditions:BTreeMap<String,f64>,
}

impl FentonKarmaBR{

        fn default_params() -> HashMap<String,f64>{
                let pnames = vec![
                "u_c".to_string(),
                "u_v".to_string(),
                "g_fi_max".to_string(),
                "tau_v1_minus".to_string(),
                "tau_v2_minus".to_string(),
                "tau_v_plus".to_string(),
                "tau_0".to_string(),
                "tau_r".to_string(),
                "k".to_string(),
                "tau_si".to_string(),
                "u_csi".to_string(),
                "tau_w_minus".to_string(),
                "tau_w_plus".to_string(),
                "Cm".to_string(),
                "V_0".to_string(),
                "V_fi".to_string()];

                let pvals = vec![0.13,
                0.04,
                4.,
                1250.,
                19.6,
                3.33,
                12.5,
                33.33,
                10.,
                29.,
                0.85,
                41.,
                870.,
                1.0,
                -85.0,
                15.0];

                let params: HashMap<_,_> = pnames
                        .into_iter()
                        .zip(pvals.into_iter())
                        .collect();

                params

        }

    fn default_initial_conditions()->BTreeMap<String,f64>{
            let icnames = vec![
                    "V".to_string(),
                    "v".to_string(),
                    "w".to_string()];
            let icvalues = vec![
                    -85.0,
                    1.0,
                    1.0];

            let ic: BTreeMap<_,_> = icnames
                    .into_iter()
                    .zip(icvalues.into_iter())
                    .collect();

            ic

    }

	pub fn new(params: Option<HashMap<String,f64>>
                ,initial_conditions:Option<BTreeMap<String,f64>>)
                 -> Self {

		let params = match params{
			Some(params) => {
				params
			}
			None =>{ 
                Self::default_params()
                        }
		};


        let initial_conditions = match initial_conditions{
                Some(initial_conditions) => {
                        initial_conditions
                }
                None =>{ 
                        Self::default_initial_conditions()
                }
        };


		FentonKarmaBR{
			params,
            initial_conditions,
		}
	}


        
}

pub trait CellModel{
    fn get_params(&mut self) -> & HashMap<String,f64>;
    fn get_init_cond(&mut self) -> & BTreeMap<String,f64>;
    fn Icurrent(&mut self,
        vr: & Array1<f64>,time:&f64) ->  Array1<f64>;
    fn F(&mut self,
        vr: & Array1<f64>,time:&f64) ->  Array1<f64>;
    fn model_eval(&mut self,
        vr: & Array1<f64>,time:&f64) ->  Array1<f64>;
    fn num_states(&mut self) -> usize;
    fn model_name(&mut self) -> String;
}


pub fn Heaviside(vm:&f64,vr:&f64) -> f64 {
    let mut a = 0.0;

    if vm >= vr{
        a = 1.0;
    }else {
        a = 0.0;
    }

    a
}


impl CellModel for FentonKarmaBR{
	fn get_params(&mut self) -> & HashMap<String,f64>{
		&self.params
	}
    
    fn get_init_cond(&mut self) -> & BTreeMap<String,f64>{
            &self.initial_conditions
    }

    fn Icurrent(&mut self,
        vr: &  Array1<f64>,time:&f64) ->  Array1<f64>{
            
            //internal variables
            
            let (vm,v,w) =(vr[0],vr[1],vr[2]);
            let mut current = Array::<f64,_>::zeros(1);
            //Assign parameters

            let u_c = self.params["u_c"];
            let g_fi_max = self.params["g_fi_max"];
            let tau_0 = self.params["tau_0"];
            let tau_r = self.params["tau_r"];
            let k = self.params["k"];
            let tau_si = self.params["tau_si"];
            let u_csi = self.params["u_csi"];
            let cm = self.params["Cm"];
            let v_0 = self.params["V_0"];
            let v_fi = self.params["V_fi"];


            let p = &Heaviside(&((vm-v_0)/(v_fi-v_0)),&u_c);
            let tau_d = cm/g_fi_max;

            // Expressions for the fast inward current component
            let jfi = -(1.0-(vm-v_0)/(v_fi-v_0))*
                (-u_c+(vm-v_0)/(v_fi-v_0))*p*v/tau_d;

            // Expressions for the Slow outward current component
            let jso = p/tau_r + (1.0-p)*(vm-v_0)/(tau_0*(v_fi-v_0));

            // Expressions for the Slow inward current component
            let jsi = -(1.0+(k*(-u_csi + (vm-v_0)/(v_fi-v_0)))
                .tanh())*w/(2.0*tau_si);
            
            // Expressions for the Membrane component
            current[0] = -(v_0-v_fi)*(jfi+jsi+jso);
            
            current
    }

    fn F(&mut self,
        vr: & Array1<f64>,time:&f64) ->  Array1<f64>{

        //internal variables
            
        let (vm,v,w) =(vr[0],vr[1],vr[2]);
        let mut f = Array::<f64,_>::zeros(2);

        //Assign parameters

        let u_c = self.params["u_c"];
        let u_v = self.params["u_v"];
        let tau_v1_minus = self.params["tau_v1_minus"];
        let tau_v2_minus = self.params["tau_v2_minus"];
        let tau_v_plus = self.params["tau_v_plus"];
        let tau_w_minus = self.params["tau_w_minus"];
        let tau_w_plus = self.params["tau_w_plus"];
        let v_0 = self.params["V_0"];
        let v_fi = self.params["V_fi"];


        let p = &Heaviside(&((vm-v_0)/(v_fi-v_0)),&u_c);

        let q = &Heaviside(&((vm-v_0)/(v_fi-v_0)),&u_v);

        // Expressions for the v gate component
        let tau_v_minus = tau_v1_minus*q + tau_v2_minus*(1.0 - q);
        f[0] = (1.0 - p)*(1.0 - v)/tau_v_minus - p*v/tau_v_plus;

        // Expressions for the w gate component
        f[1] = (1.0 - p)*(1.0 - w)/tau_w_minus - p*w/tau_w_plus;

        f

    }

    fn model_eval(&mut self,
        vr: & Array1<f64>,time:&f64) ->  Array1<f64>{
        let i_current = - self.Icurrent(vr,time);
        let f_states = self.F(vr,time);

        let f = stack![Axis(0),
        i_current,
        f_states];

        f
    }

    fn num_states(&mut self ) -> usize {
        2
    }
    
    fn model_name(&mut self) -> String{
        String::from("Fenton Karma 1998 altered cardiac cell model")
    }
	
}

pub struct Stimulus {
    pub start:f64,
    pub duration:f64,
    pub amplitude:f64,
    pub period:f64,
    pub time:f64,
    pub expression: Box<dyn Fn(&f64,&f64,&f64,&f64,&f64) ->f64>
}

impl Stimulus {
    fn new(start:f64,duration:f64,amplitude:f64,
        period:f64,expression:Box<dyn Fn(&f64,&f64,&f64,&f64,&f64)->f64> ) -> Self {
        Stimulus{
            start,
            duration,
            amplitude,
            period,
            expression,
            time:0.0,
        }
    }

    fn eval(&self,time:&f64) -> f64{
        (self.expression)(
            &self.start,
            &self.duration,
            &self.amplitude,
            &self.period,
            time)
    }

}

pub struct Field {
    pub n :  Array1<f64>, //t(n)
    pub n_1 : Array1<f64>,  //t(n+1)
}

impl Field {
    fn update(&mut self) {
         self.n = &self.n_1 + 0.0;
    }
}

pub struct ODEsolver<T> {
    cellmodel: T,
    time: f64,
    params: HashMap<String,f64>,
    pub vr :Field,
    Is: Stimulus,
    //times
    dt:f64,
    pub t0 :f64,
    pub t1: f64,
    t0_:f64,
    t1_:f64,
}

pub fn Inverse3x3(m:&Array2<f64>)->Array2<f64>{
    let (a,b,c,d,e,f,g,h,i) = (m[[0,0]],m[[0,1]],m[[0,2]],
                                m[[1,0]],m[[1,1]],m[[1,2]],
                                m[[2,0]],m[[2,1]],m[[2,2]]);
    
    let det = 1.0/(a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g));

    let coff = arr2(&[[e*i-f*h,c*h-b*i,b*f-c*e],
                        [f*g-d*i,a*i-c*g,c*d-a*f],
                        [d*h-e*g,b*g-a*h,a*e-b*d]]);
    det*coff

}

pub fn Norm_v(v:&Array1<f64>)->f64{
    let sumvsq:f64 = v.iter().map(|x| x*x).sum();
    sumvsq.sqrt()
}

impl<T:CellModel> ODEsolver<T> {
    pub fn new(mut model:T,time:f64,Is:Stimulus
        ,params: Option<HashMap<String,f64>>) -> Self{
        let params = match params{
            Some(params) => {
                params
            }
            None =>{ 
                Self::default_params()
                        }
        };

        

        let in_cond:Array1<f64> = model
        .get_init_cond()
        .values()
        .cloned()
        .collect();

        let mut vr = Field{
            n : in_cond.clone(),
            n_1:in_cond,
        };

        ODEsolver{
            cellmodel: model,
            time: time,
            params: params,
            vr: vr,
            Is:Is,
            //times
            dt:0.0,
            t0 :0.0,
            t1: 0.0,
            t0_:0.0,
            t1_:0.0,
        }
    }

    fn default_params()->HashMap<String,f64>{
        let mut params = HashMap::new();

        params.insert("theta".to_string(),0.5);
        params
    }



    fn calc_DR(&mut self,interval:&(f64,f64)) -> Array2<f64> {
        // computing tangent [DR] using finite differences.
        
        let theta = self.params["theta"];
        let (t0,t1) = interval;
        let dt = t1-t0;
        let num_states = self.cellmodel.num_states();

        let delta = Array::<f64,_>::eye(&num_states+1)*0.001;
        let dt_der = Array::<f64,_>::eye(&num_states+1)*(1.0/dt);

        let vr_mid = &self.vr.n_1*theta + &((1.0-theta)*&self.vr.n);
        let mut dr = Array::<f64,_>
            ::zeros((&num_states+1,&num_states+1));

        for j in 0..num_states+1{
            let vr_h = &vr_mid - &delta.slice(s![..,j]); 
            let vr_h_1 = &vr_mid + &delta.slice(s![..,j]);

            let mut slice = dr.slice_mut(s![..,j]);

            slice += &(self.cellmodel.model_eval(&vr_h_1,&self.time)-
            self.cellmodel.model_eval(&vr_h,&self.time)).view();
          
        }
        let dr = dt_der - dr;

        dr


    }

    fn solution_fields(&self) -> &Field {
        &self.vr
    }

    fn calc_R(&mut self,interval:&(f64,f64))->Array1<f64>{
        let (t0,t1) = interval;
        let dt = t1-t0;
        let theta = self.params["theta"];

        let vr_mid = &self.vr.n_1*theta + &((1.0-theta)*&self.vr.n);

        let mut f = self.cellmodel.model_eval(&vr_mid,&self.time);

        f[0] += self.Is.eval(&self.time);

        let res = (&self.vr.n_1-&self.vr.n)/dt - &f;

        res
    }
    
    pub fn step_ode(&mut self,interval:&(f64,f64)){
        let (t0,t1) = interval;
        let theta = self.params["theta"];
        let t = t0+theta*t1;
        self.time = t;

        let res = self.calc_R(interval);

        let (a_tol,r_tol) = (1.0e-7,1.0e-10);
        // let mut u_inc = Array::<f64,_>::
            // zeros(&self.cellmodel.num_states()+1);
        let  (mut niter, mut eps) = (0,1.0);


        while (eps > 1.0e-10 && niter <= 20) {
            niter += 1;
           
            let mut dr = self.calc_DR(interval);
            let mut res = - self.calc_R(interval);


            let u_inc = Inverse3x3(&dr).dot(&res);
            eps = Norm_v(&u_inc);
            self.vr.n_1 =  &self.vr.n_1 + &u_inc;

            // println!("niter: {:?}\n",niter );
            // println!("vr_n_1: {:?}\n",&self.vr.n_1 );
            // println!("eps: {:?}\n",eps );

            if (niter >= 20){
                println!("max number of 
                    iterations reached norm(u_inc):{}\n",eps);
                // println!("res : {:?}\n dr:{:?}", &res,&dr);
            }
        }
    }
    pub fn set_times(&mut self,interval:(f64,f64),dt:f64){
        self.dt = dt;
        self.t0 = interval.0;
        self.t1 = interval.0;
        self.t0_= interval.0;
        self.t1_= interval.1; 
    }
}

impl<T:CellModel> Iterator for ODEsolver<T>{
    type Item = (f64,f64);

    fn next(& mut self) -> Option<Self::Item> {
        
        self.t0 = self.t1;
        if (self.t0 + self.dt) <= self.t1_{
            self.t1 = self.t0 + self.dt;
            self.step_ode(&(self.t0,self.t1));
            self.vr.update();
            return Some((self.t1,self.vr.n_1[0]));
        }else{
            None  
        }
        
    }
}