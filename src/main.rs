extern crate ndarray;

use ndarray::prelude::*;
use cell_solver::{FentonKarmaBR,Stimulus,ODEsolver};
use itertools::Itertools;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use plotters::prelude::*;


fn main() {
	
	let mut in_cond = BTreeMap::new();

	in_cond.insert("V".to_string(),-85.0);
	in_cond.insert("v".to_string(),1.0);
	in_cond.insert("w".to_string(),1.0);

	let mut fk = FentonKarmaBR::new(None,Some(in_cond));
	fn hello(a:&f64,b:&f64,c:&f64,d:&f64,e:&f64)->f64{
		0.0
	}

	let stim = Stimulus{
		start:0.0,
		duration:1.0,
		amplitude:0.0,
		period:0.0,
		expression:Box::new(hello),
		time:1.0,
	};

	let mut solver = ODEsolver::new(fk,0.0,stim,None);
	solver.set_times((0.,300.),0.1);
	

	let mut output_file= OpenOptions::new()
	.append(true)
	.open("outputs/output_fk.txt")
	.expect("cannot open file");

	let mut vres = Vec::new();

	for i in 0..(300./0.2) as usize{
		solver.next();
		let mut res = match solver.next() {
			Some(expr) => expr,
			None => (0.0,0.0),
		};
		vres.push(res);
	}

	// plot
	
	let root_area = BitMapBackend::new("images/voltage.png", (1200, 800))
    .into_drawing_area();
	root_area.fill(&WHITE).unwrap();

	let mut ctx = ChartBuilder::on(&root_area)
	.set_label_area_size(LabelAreaPosition::Left, 40)
	.set_label_area_size(LabelAreaPosition::Bottom, 40)
	.caption("Voltage F-K model", ("sans-serif", 20))
	.build_cartesian_2d(0.0..300.0, -85.0..40.0)
	.unwrap();

	ctx.configure_mesh().draw().unwrap();

	ctx.draw_series(
	LineSeries::new(vres.iter().map(|(x,y)| (*x,*y)),&RED)
	).unwrap();

}


//     pub fn s1_stimulus(&self,&t)->f64{
        
//         let amp =  self.amplitude
//         if (t-
//             ((t/self.period)
//             .floor())
//             *self.period
//              >= self.duration + self.start){
//             if (t-
//                 ((t/self.period)
//                 .floor())
//                 *self.period
//                  <= self.duration + self.start) {
//                 self.amplitude
//             }else{
//                 0.0
//             }
//         }else {
//             0.0
//         }
//         self.time = t;
//         amp
//     }
