

use std::thread;
use std::sync::mpsc;
use std::time::Duration;
use std::sync::{Arc, Mutex};
use std::path::Path;

extern crate nalgebra_glm as glm;
use rand::prelude::*;

extern crate piston_window;
extern crate image as im;
extern crate vecmath;
extern crate gfx;
extern crate gfx_core;
extern crate gfx_device_gl;

use piston_window::*;
use vecmath::*;

use std::io;


#[derive(Debug)]
struct Image {
    width: i32,
    height: i32,
    buffer: Vec<u8>
}

impl Image {
    pub fn new( width: i32, height: i32) -> Image {
        let buffer_size: usize = (width*height*3) as usize;
        Image { width, height, buffer: vec![0 as u8;buffer_size] }
    }

    pub fn set_pixel( &mut self, x: i32, y: i32, color: glm::Vec3) {
        let y = self.height - y - 1; // flip the y axis to match the book
        let pixel_index: usize = (self.width * y + x) as usize;

        let r_index = pixel_index*3;
        let g_index = pixel_index*3+1;
        let b_index = pixel_index*3+2;
        
        self.buffer[r_index] = color.x as u8;
        self.buffer[g_index] = color.y as u8;
        self.buffer[b_index] = color.z as u8;
    }

    pub fn write(&self) {
        // For reading and opening files
        use std::path::Path;
        use std::fs::File;
        use std::io::BufWriter;
        // To use encoder.set()
        use png::HasParameters;

        let path = Path::new(r"rt1.png");
        let file = File::create(path).unwrap();
        let ref mut w = BufWriter::new(file);

        let mut encoder = png::Encoder::new(w, self.width as u32, self.height as u32);
        encoder.set(png::ColorType::RGB).set(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();

        writer.write_image_data(&self.buffer).unwrap(); // Save
    }
}

#[derive(Debug)]
pub struct Ray {
    a: glm::Vec3,
    b: glm::Vec3,
}

impl Ray {
    pub fn new( a: glm::Vec3, b: glm::Vec3 ) -> Ray {
        Ray { a, b }
    }

    pub fn origin(&self) -> &glm::Vec3 {
        &self.a
    }

    pub fn direction(&self) -> &glm::Vec3 {
        &self.b
    }

    pub fn point_at_parameter(&self, t: f32 ) -> glm::Vec3 {
        self.a + t * self.b
    }
}

fn sky_color( direction: glm::Vec3 ) -> glm::Vec3 {
    let t: f32 = 0.5 * (direction.y + 1.0);
    let c: glm::Vec3  = (1.0-t)*glm::vec3(1.0,1.0,1.0) + t*glm::vec3(0.5,0.7,1.0);
    c
}

fn get_rand_jitter() -> f32 {
    let mut rng = rand::thread_rng();

    let rand_int = rng.gen::<u32>();
    let rand_float: f32 = rand_int as f32/std::u32::MAX as f32;

    rand_float
}

fn gamma_correct( color: glm::Vec3 ) -> glm::Vec3 {
    glm::vec3( color.x.sqrt(), color.y.sqrt(), color.z.sqrt() )
}



fn get_color_from_ray( ray: &Ray, hitables: &Vec<Arc<Hitable>>, depth: u32 ) -> glm::Vec3 {

    let mut color: glm::Vec3 = glm::vec3(0.0,0.0,0.0);
    let mut rec: HitRecord = HitRecord::new();

    if hit_list(&hitables, &ray, 0.001, std::f32::MAX, &mut rec) {

        let mut scattered: Ray = Ray::new( glm::vec3(0.0,0.0,0.0), glm::vec3(0.0,0.0,0.0) );
        let mut attenuation: glm::Vec3 = glm::vec3(0.0,0.0,0.0);

        if depth < 50 && scatter_material( rec.material, ray, &rec, &mut attenuation, &mut scattered ) {
            //println!("scattered ray {:?}",scattered.direction().normalize() );
            let r_color = get_color_from_ray( &scattered, &hitables, depth+1 );
            return glm::vec3( attenuation.x*r_color.x, attenuation.y*r_color.y, attenuation.z*r_color.z );
        }
    }
    else
    {
        //println!("SKY {:?}",ray.direction().normalize() );
        color = sky_color(ray.direction().normalize());
    }

    color
}

pub struct GraphicsWindow {
    width: u32,
    height: u32,
    opengl: OpenGL,
    window: piston_window::PistonWindow,
    canvas: im::ImageBuffer<im::Rgba<u8>, std::vec::Vec<u8>>,
    draw: bool,
    window_scale: f64,
}

impl GraphicsWindow {
    fn new( width: u32, height: u32 ) -> GraphicsWindow {
        let window_width = 1024;
        let window_scale = window_width as f64 / width as f64;

        let opengl = OpenGL::V3_2;
        let scaled_window_width = window_width;
        let scaled_window_height = (height as f64 * window_scale) as u32;
        let mut window: PistonWindow =
           WindowSettings::new("Rust Raytracer", (scaled_window_width, scaled_window_height))
            .exit_on_esc(true)
            .opengl(opengl)
            .build()
            .unwrap();

        let canvas = im::ImageBuffer::new(width, height);
        let draw = false;
        let texture: G2dTexture = Texture::from_image(
                &mut window.factory,
                &canvas,
                &TextureSettings::new()
            ).unwrap();
        GraphicsWindow { width, height, opengl, window, canvas, draw: false, window_scale }
    
    }

    fn set_pixel( &mut self, x: u32, y: u32, color: &glm::Vec3 ) {
        self.canvas.put_pixel(x, self.height - y - 1, im::Rgba([color.x as u8, color.y as u8, color.z as u8, 255]));
    }

    fn set_pixel_block( &mut self, data: &RenderBlockInputData ) {
        //println!("Compositing block {}, {}", data.window_offset_upperleft.0, data.window_offset_upperleft.1 );

        for x in 0..data.block_size*data.pixel_scale {
            for y in 0..data.block_size*data.pixel_scale {
                let pix_num = (y/data.pixel_scale)*data.block_size+(x/data.pixel_scale);
                let color = data.output_buffer[pix_num as usize];

                self.canvas.put_pixel( data.pixel_offset_upperleft.0+x, self.height - (data.pixel_offset_upperleft.1+y) - 1, im::Rgba([color.x as u8, color.y as u8, color.z as u8, 255]) );
            }
        }

    }

    fn draw(&mut self) {
        while let Some(e) = self.window.next() {
            if let Some(_) = e.render_args() {
                //self.texture.update(&mut self.window.encoder, &self.canvas).unwrap();
                let texture: G2dTexture = Texture::from_image(
                    &mut self.window.factory,
                    &self.canvas,
                    &TextureSettings::new()
                ).unwrap();
                let mut transform = math::translate([0.0,0.0]);
                transform[0][2] = -1.0;
                transform[1][2] = 1.0;
                transform = transform.scale(0.002*self.window_scale,-0.004*self.window_scale);
                // [[0.016666666666666666, 0.0, -1.0], [0.0, -0.02, 1.0]]
                self.window.draw_2d(&e, |c, g| {
                    clear([1.0; 4], g);
                    image(&texture, transform, g);
                
                });
                break;
            }
        }
    }

    pub fn save_image(&self, image_name: &str) {
        // For reading and opening files
        use std::path::Path;
        use std::fs::File;
        use std::io::BufWriter;
        // To use encoder.set()
        use png::HasParameters;

        let path = Path::new(image_name);
        let file = File::create(path).unwrap();
        let ref mut w = BufWriter::new(file);

        let mut encoder = png::Encoder::new(w, self.width as u32, self.height as u32);
        encoder.set(png::ColorType::RGB).set(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();

        let buffer_size = self.width * self.height * 3;
        let mut buffer = vec![0 as u8;buffer_size as usize];

        // copy from the canvas to our image buffer
        for x in 0..self.width {
            for y in 0..self.height {
                let pixel = self.canvas.get_pixel( x, y );

                let pixel_index: usize = (self.width * y + x) as usize;

                let r_index = pixel_index*3;
                let g_index = pixel_index*3+1;
                let b_index = pixel_index*3+2;
                
                buffer[r_index] = pixel[0] as u8;
                buffer[g_index] = pixel[1] as u8;
                buffer[b_index] = pixel[2] as u8;
            }
        }

        writer.write_image_data(&buffer).unwrap(); // Save
    }

}

pub fn random_in_unit_sphere() -> glm::Vec3 {
    let mut p: glm::Vec3 = glm::vec3(0.0,0.0,0.0);

    loop {
        p = 2.0 * glm::vec3(get_rand_jitter(), get_rand_jitter(), get_rand_jitter() ) - glm::vec3(1.0,1.0,1.0);
        if glm::length2( &p ) < 1.0 {
            break;
        }
    }

    p
}

#[derive(Debug)]
pub struct RenderBlockInputData {
    image_size: (u32, u32),
    block_size: u32,
    pixel_scale: u32,
    pixel_offset_upperleft: (u32,u32),
    window_offset_upperleft: (f32,f32),
    window_block_size: (f32,f32),
    output_buffer: Vec<glm::Vec3>,
}

pub struct RenderThreadDataGenerator {
    block_size: u32,
    nx: u32,
    ny: u32,

    pixel_scale: u32,
    start_pixel_scale: u32,
    end_pixel_scale: u32,
    current_x: u32,
    current_y: u32,
}

impl RenderThreadDataGenerator {
    fn new( block_size: u32, nx: u32, ny: u32, start_pixel_scale: u32, end_pixel_scale: u32 ) -> RenderThreadDataGenerator {
        RenderThreadDataGenerator { 
            block_size, nx, ny, start_pixel_scale, end_pixel_scale,
            pixel_scale: start_pixel_scale, current_x: 0, current_y: 0,
         }
    }

    fn get_next_block(&mut self) -> RenderBlockInputData {
        let empty_block = RenderBlockInputData {
            image_size: (self.nx,self.ny),
            block_size: 0,
            pixel_scale: 0,
            pixel_offset_upperleft: (0,0),
            window_offset_upperleft: (0.0,0.0),
            window_block_size: (0.0,0.0),
            output_buffer: Vec::new(),
        };

        if self.pixel_scale == 0 {
            return empty_block;
        }

        let pixel_block_size = self.pixel_scale * self.block_size;

        let block = self.gen_block();

        self.current_x+=1;
        if self.current_x >= self.nx/pixel_block_size {
            self.current_x=0;
            self.current_y+=1;
            if self.current_y>=self.ny/pixel_block_size {
                self.current_y=0;
                self.pixel_scale/=2;
                if self.pixel_scale<self.end_pixel_scale {
                    self.pixel_scale = 0;

                    return empty_block;
                }
            }
        }

        block
    }

    fn gen_block(&self) -> RenderBlockInputData {
        let pixel_block_size = self.pixel_scale * self.block_size;

        RenderBlockInputData {
            image_size: (self.nx,self.ny),
            block_size: self.block_size,
            pixel_scale: self.pixel_scale,
            pixel_offset_upperleft: (self.current_x*pixel_block_size, self.current_y*pixel_block_size),
            window_offset_upperleft: ((self.current_x as f32 * pixel_block_size as f32)/self.nx as f32,((self.current_y as f32 * pixel_block_size as f32)/self.ny as f32)),
            window_block_size: (pixel_block_size as f32 / self.nx as f32, pixel_block_size as f32 / self.ny as f32),
            output_buffer: Vec::new(),
        }
    }
}

fn render_proc( data: RenderBlockInputData, hitables: Vec<Arc<Hitable>> ) -> RenderBlockInputData {
    println!("Rendering block {:?}", data);

    let ns = 150;
    let lookfrom = glm::vec3( 20.0, 2.2, 4.0 );
    let lookat = glm::vec3( 0.0, 0.5, 0.0 );
    let vup = glm::vec3( 0.0, 1.0, 0.0 );
    let vfov = 10.0;
    let aspect = 2.0;
    let aperature = 0.2;
    let focus_dist = (lookfrom-lookat).magnitude();
    let camera = Camera::new( lookfrom, lookat, vup, vfov, aspect, aperature, focus_dist );

    let mut ret_block = RenderBlockInputData { image_size: data.image_size, block_size: data.block_size, pixel_scale: data.pixel_scale, pixel_offset_upperleft: data.pixel_offset_upperleft, window_offset_upperleft: data.window_offset_upperleft, window_block_size: data.window_block_size, output_buffer: Vec::new(), };
    ret_block.output_buffer.resize((data.block_size*data.block_size)as usize, glm::vec3(0.0,0.0,0.0));

    if data.block_size == 0 {
        // we're done.
        return ret_block;
    }

    for y in 0..data.block_size {
        for x in 0..data.block_size {

            let mut color: glm::Vec3 = glm::vec3(0.0,0.0,0.0);

            for _s in 0..ns {
                let mut u: f32 = data.window_offset_upperleft.0 + data.window_block_size.0 * (x as f32/data.block_size as  f32);
                let mut v: f32 = data.window_offset_upperleft.1 + data.window_block_size.1 * (y as f32/data.block_size as  f32);

                u += get_rand_jitter()/data.image_size.0 as f32;
                v += get_rand_jitter()/data.image_size.1 as f32;

                let ray: Ray = camera.get_ray( u, v );

                //println!("");
                //println!("new ray {:?}",ray.direction().normalize() );

                color = color + get_color_from_ray( &ray, &hitables, 0 );
            }

            color = gamma_correct( color/ns as f32 );

            let ir = color.x as f32 * 255.9;
            let ig = color.y as f32 * 255.9;
            let ib = color.z as f32 * 255.9;

            ret_block.output_buffer[(y*data.block_size+x)as usize] = glm::vec3( ir, ig, ib );
        }
    }

    ret_block
}

pub fn build_scene( hitables: &mut Vec<Arc<Hitable>> ) {
    hitables.push( Arc::new( Sphere::new( glm::vec3(0.0, -1000.0, 0.0), 1000.0,      MaterialData::Lambertian( LambertianData { albedo: glm::vec3(0.5, 0.5, 0.5) } ) ) ) );

    for a in -11..11 {
        for b in -11..11 {
            let chose_mat = get_rand_jitter();
            let center = glm::vec3( a as f32+0.9*get_rand_jitter(), 0.2, b as f32+0.9*get_rand_jitter() );
            if ( center-glm::vec3(4.0,0.2,0.0) ).magnitude() > 0.9 {
                if chose_mat < 0.9 {
                    hitables.push( Arc::new( Sphere::new( center, 0.2,      MaterialData::Lambertian( LambertianData { albedo: glm::vec3(get_rand_jitter()*get_rand_jitter(), get_rand_jitter()*get_rand_jitter(), get_rand_jitter()*get_rand_jitter()) } ) ) ) );
                }
                else if chose_mat < 0.95
                {
                    hitables.push( Arc::new( Sphere::new( center, 0.2,      MaterialData::Metal( MetalData { albedo: glm::vec3( 0.5 * (1.0 + get_rand_jitter() ),0.5 * (1.0 + get_rand_jitter() ),0.5 * (1.0 + get_rand_jitter() )) } ) ) ) );
                }
                else
                {
                    hitables.push( Arc::new( Sphere::new( center, 0.2,      MaterialData::Dielectric( DielectricData { index_of_refraction: 1.5 } ) ) ) );
                }
            }
        }
    }    

    hitables.push( Arc::new( Sphere::new( glm::vec3(0.0, 1.0, 0.0), 1.0,     MaterialData::Dielectric( DielectricData { index_of_refraction: 1.5 } ) ) ) );
    hitables.push( Arc::new( Sphere::new( glm::vec3(-4.0, 1.0, 0.0), 1.0,      MaterialData::Lambertian( LambertianData { albedo: glm::vec3(0.4, 0.2, 0.1) } ) ) ) );
    hitables.push( Arc::new( Sphere::new( glm::vec3(4.0, 1.0, 0.0), 1.0,      MaterialData::Metal( MetalData { albedo: glm::vec3(0.7, 0.6, 0.5) } ) ) ) );

}

fn render_scene() { 

    let block_size = 16; 

    let nx = 1024;
    let ny = 512;

    let start_pixel_scale = ny/block_size;
    let end_pixel_scale = 1;

    let mut block_gen = RenderThreadDataGenerator::new( block_size, nx, ny, start_pixel_scale, end_pixel_scale );

    let max_threads = 12;

    let mut window = GraphicsWindow::new( nx, ny );

    let mut hitables: Vec<Arc<Hitable>> = Vec::new();
    //hitables.push( Arc::new( Sphere::new( glm::vec3(0.0, 0.0, -1.0), 0.5,      MaterialData::Lambertian( LambertianData { albedo: glm::vec3(0.8, 0.3, 0.3) } ) ) ) );
    //hitables.push( Arc::new( Sphere::new( glm::vec3(0.0, -100.5, -1.0), 100.0, MaterialData::Lambertian( LambertianData { albedo: glm::vec3(0.2, 0.8, 0.8) } ) ) ) );
    //hitables.push( Arc::new( Sphere::new( glm::vec3(1.0, 0.0, -1.0), 0.5,      MaterialData::Metal( MetalData { albedo: glm::vec3(0.8, 0.6, 0.2) } ) ) ) );
    //hitables.push( Arc::new( Sphere::new( glm::vec3(-1.0, 0.0, -1.0), 0.5,     MaterialData::Dielectric( DielectricData { index_of_refraction: 1.5 } ) ) ) );

    build_scene( &mut hitables );

    window.draw();  
    
    let (tx, rx) = mpsc::channel::<RenderBlockInputData>();

    let mut active_threads = 0;

    for t in 0..max_threads {
        let next_block = block_gen.get_next_block();
        let new_tx = mpsc::Sender::clone(&tx);

        let hitables_clone: Vec<Arc<Hitable>> = hitables.clone();

        thread::spawn( move || {
            let finished_block = render_proc( next_block, hitables_clone );
            new_tx.send( finished_block ).unwrap();
        });
        active_threads += 1;
    }

    for received in rx {
        window.set_pixel_block( &received );
        window.draw();
        active_threads -= 1;

        if active_threads == 0 {
            break;
        }

        let next_block = block_gen.get_next_block();

        if next_block.block_size != 0 {

            let new_tx = mpsc::Sender::clone(&tx);

            let hitables_clone: Vec<Arc<Hitable>> = hitables.clone();

            thread::spawn( move || {
                let finished_block = render_proc( next_block, hitables_clone );
                match new_tx.send( finished_block ) {
                Ok(_)  => {},
                Err(_) => {},
            }
            });

            active_threads += 1;
        }
    }

    let save_filename = get_image_filename();

    println!("Render complete. Saving image {}.", save_filename);

    window.save_image(&save_filename[..]);

    loop {
        window.draw();
    }

    io::stdin().read_line(&mut String::new()).unwrap();

}

pub fn get_image_filename() -> String {
     let mut base_index = 0;

    loop {
        let base = "raytrace-";
        let suffix = ".png";
        let mut filename: String = String::new();

        filename.push_str(base);
        filename.push_str(&base_index.to_string()[..]);
        filename.push_str(suffix);

        if Path::new(&filename[..]).exists() {
            //println!("{} already exists", filename );
            base_index +=1;
        }
        else
        {
            return filename;
        }
    }
}

#[derive (Copy, Clone)]
pub struct LambertianData {
    albedo: glm::Vec3
}

#[derive (Copy, Clone)]
pub struct MetalData {
    albedo: glm::Vec3
}

#[derive (Copy, Clone)]
pub struct DielectricData {
    index_of_refraction: f32
}

#[derive (Copy, Clone)]
pub enum MaterialData {
    Lambertian ( LambertianData ),
    Metal ( MetalData ),
    Dielectric ( DielectricData ),
}

pub fn lambertian_scatter( mat_params: LambertianData, r_in: &Ray, rec: &HitRecord, attenuation: &mut glm::Vec3, scattered: &mut Ray ) -> bool {
    let target = rec.p + rec.normal + random_in_unit_sphere();
    *scattered = Ray::new( rec.p, target-rec.p );
    *attenuation = mat_params.albedo;
    return true;
}

pub fn reflect_vec( v: glm::Vec3, n: glm::Vec3 ) -> glm::Vec3 {
    return v - 2.0*v.dot(&n)*n;
}

pub fn metal_scatter( mat_params: MetalData, r_in: &Ray, rec: &HitRecord, attenuation: &mut glm::Vec3, scattered: &mut Ray ) -> bool {
    let reflected = reflect_vec( r_in.direction().normalize(), rec.normal );
    *scattered = Ray::new( rec.p, reflected );
    *attenuation = mat_params.albedo;
    return true;
}

pub fn refract_vec( v: glm::Vec3, n: glm::Vec3, ni_over_nt: f32, refracted: &mut glm::Vec3 ) -> bool {

    let uv = v.normalize();
    let dt = uv.dot( &n );
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0-dt*dt);
    if discriminant > 0.0 {
        *refracted = ni_over_nt * (uv - n*dt) - n * discriminant.sqrt();
        //println!("v {:?} n {:?} ni_over_nt {:?} refracted {:?} ", uv, n, ni_over_nt, *refracted);
        return true;
    }
    return false;
}

pub fn schlick( cosine: f32, ref_idx: f32 ) -> f32 {
    let mut r0 = (1.0-ref_idx) / (1.0+ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0-r0) * (1.0-cosine).powf(5.0);
}

pub fn dielectric_scatter( mat_params: DielectricData, r_in: &Ray, rec: &HitRecord, attenuation: &mut glm::Vec3, scattered: &mut Ray ) -> bool {
    let outward_normal: glm::Vec3;
    let reflected = reflect_vec( *r_in.direction(), rec.normal );
    let ni_over_nt: f32;
    *attenuation = glm::vec3(1.0,1.0,1.0);
    let mut refracted = glm::vec3(1.0,1.0,1.0);
    let reflect_prob: f32;
    let cosine: f32;

    //println!("test {:?} {:?} {:?} ", r_in.direction().normalize(), rec.normal, r_in.direction().normalize().dot( &rec.normal ) );

    if r_in.direction().normalize().dot( &rec.normal.normalize() ) > 0.0 {
        // ray leaving object
        //println!("leaving obj");
        outward_normal = -rec.normal;
        ni_over_nt = mat_params.index_of_refraction;
        cosine = mat_params.index_of_refraction * r_in.direction().dot( &rec.normal ) / r_in.direction().magnitude();
    }
    else 
    {
        // ray entering object from air
        //println!("entering obj");
        outward_normal = rec.normal;
        ni_over_nt = 1.0 / mat_params.index_of_refraction;
        cosine = -1.0 * r_in.direction().dot( &rec.normal ) / r_in.direction().magnitude();
    }

    if refract_vec( *r_in.direction(), outward_normal, ni_over_nt, &mut refracted ) {
        // steep angles have a chance to reflect
        reflect_prob = schlick( cosine, mat_params.index_of_refraction );
        //println!("refracting");
    }
    else 
    {
        reflect_prob = 1.0;
    }

    if get_rand_jitter() < reflect_prob {
        //println!("reflected");
        *scattered = Ray::new( rec.p, reflected );
    }
    else 
    {
        //println!("refracted");
        *scattered = Ray::new( rec.p, refracted );
    }

    true
}

pub fn scatter_material( material: MaterialData, r_in: &Ray, rec: &HitRecord, attenuation: &mut glm::Vec3, scattered: &mut Ray ) -> bool {
    match material {
        MaterialData::Lambertian( mat_params ) => {
            //println!("lamb");
            return lambertian_scatter(mat_params, r_in, rec, attenuation, scattered );
        },
        MaterialData::Metal( mat_params ) => {
            //println!("metal");
            return metal_scatter( mat_params, r_in, rec, attenuation, scattered );
        },
        MaterialData::Dielectric( mat_params ) => {
            //println!("diel");
            return dielectric_scatter( mat_params, r_in, rec, attenuation, scattered );
        },
    }
}

pub struct HitRecord {
    t: f32,
    p: glm::Vec3,
    normal: glm::Vec3,
    material: MaterialData,
}

impl HitRecord {
    pub fn new() -> HitRecord {
        HitRecord { t: 0.0, p: glm::vec3(0.0,0.0,0.0), normal: glm::vec3(0.0,0.0,0.0), material: MaterialData::Lambertian( LambertianData { albedo: glm::vec3(0.0,0.0,0.0) } ) }
    }
}

pub trait Hitable: Send + Sync  {
    fn hit( &self, ray: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord ) -> bool;
}

pub struct Sphere {
    center: glm::Vec3,
    radius: f32,
    material: MaterialData,
}

impl Sphere {
    fn new( center: glm::Vec3, radius: f32, material: MaterialData ) -> Sphere {
        Sphere { center, radius, material }
    }
}

impl Hitable for Sphere {
    fn hit<'a>( &self, ray: &'a Ray, t_min: f32, t_max: f32, rec: &'a mut HitRecord ) -> bool {
        let oc: glm::Vec3 = ray.origin() - self.center;
        let a: f32 = ray.direction().dot(ray.direction());
        let b: f32 = oc.dot( ray.direction() );
        let c: f32 = oc.dot( &oc ) - self.radius*self.radius;
        let discriminant: f32 = b*b - a*c;

        if discriminant > 0.0 {
            let temp = (-b - (b*b-a*c).sqrt())/a;

            if temp < t_max && temp > t_min {
                rec.t = temp;
                rec.p = ray.point_at_parameter( rec.t );
                rec.normal = (rec.p - self.center) / self.radius;
                rec.material = self.material;
                return true;
            }
            
            let temp = (-b + (b*b-a*c).sqrt())/a;
            if temp < t_max && temp > t_min {
                rec.t = temp;
                rec.p = ray.point_at_parameter(rec.t);
                rec.normal = (rec.p - self.center) / self.radius;
                rec.material = self.material;
                return true;
            }
        }

        // no hit
        false
    }
}

pub struct HitableList {
    hitables: Vec<Box<Hitable>>,
}

impl HitableList {
    fn hit( &self, ray: Ray, t_min: f32, t_max: f32, rec: &mut HitRecord ) -> bool {
        //let temp_rec_ref: &mut HitRecord = &mut temp_rec;
        let mut hit_anything: bool = false;
        let mut closest_so_far: f32 = t_max;

        for hitable in &self.hitables {
            let mut temp_rec: HitRecord = HitRecord::new();
            if hitable.hit( &ray, t_min, t_max, &mut temp_rec ) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                *rec = temp_rec;
            }
        }

        hit_anything
    }

    fn add( &self, hitable: &mut Hitable ) {
        //self.hitables.push( Box::new(hitable) );
    }
}

fn hit_list(hit_list: &Vec<Arc<Hitable>>, ray: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord ) -> bool {
    let mut hit_anything: bool = false;
    let mut closest_so_far: f32 = t_max;

    for hitable in hit_list {
        let mut temp_rec: HitRecord = HitRecord::new();
        if hitable.hit( &ray, t_min, closest_so_far, &mut temp_rec ) {
            hit_anything = true;
            *rec = temp_rec;
            closest_so_far = rec.t;
        }
    }

    hit_anything
}

pub struct Camera {
    origin: glm::Vec3,
    lower_left_corner: glm::Vec3,
    horizontal: glm::Vec3,
    vertical: glm::Vec3,
    lens_radius: f32,
    u: glm::Vec3,
    v: glm::Vec3,
    w: glm::Vec3,
}

pub fn random_in_unit_disk() -> glm::Vec3 {
    let mut p: glm::Vec3;
    loop {
        p = 2.0 * glm::vec3( get_rand_jitter(), get_rand_jitter(), 0.0) - glm::vec3(1.0,1.0,0.0);
        if p.dot( &p ) < 1.0 {
            return p;
        }
    }
}

impl Camera {
    fn new( lookfrom: glm::Vec3, lookat: glm::Vec3, vup: glm::Vec3, vfov: f32, aspect: f32, aperature: f32, focus_dist: f32 ) -> Camera {
        let theta = vfov * std::f32::consts::PI / 180.0;
        let half_height = ( theta/2.0 ).tan();
        let half_width = aspect * half_height;

        let w = (lookfrom - lookat).normalize();
        let u = vup.cross( &w ).normalize();
        let v = w.cross( &u );

        Camera {
            lower_left_corner: lookfrom - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w,
            horizontal: 2.0*half_width*focus_dist*u,
            vertical: 2.0*half_height*focus_dist*v,
            origin: lookfrom,
            lens_radius: aperature / 2.0,
            w, u, v,
        }
    }

    fn get_ray( &self, s: f32, t: f32 ) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;
        Ray { a: self.origin + offset, b: self.lower_left_corner + s*self.horizontal + t*self.vertical - self.origin - offset }
    }
}

fn main() {
    render_scene();
} 

