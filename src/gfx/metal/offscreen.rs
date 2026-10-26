use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::{Instant, Duration};
use objc2::{rc::{autoreleasepool, Retained}, runtime::ProtocolObject};
use objc2_metal::{ MTLCreateSystemDefaultDevice, MTLDevice, MTLDeviceLocation };
use skia_safe::{ImageInfo, Surface};
use skia_safe::gpu::{ surfaces, Budgeted, DirectContext, SurfaceOrigin };
use serde_json::{json, Value};

use crate::gfx::{page::ExportOptions, engine::render_thread};
use super::make_direct_context;

thread_local!( static MTL_CONTEXT: RefCell<Option<MetalContext>> = const { RefCell::new(None) }; );
static MTL_CONTEXT_LIFESPAN:Duration = Duration::from_millis(2500); // rebuild is cheap, so use a short lifespan
static MTL_STATUS: OnceLock<Value> = OnceLock::new();

//
// Offscreen rendering
//
pub struct MetalEngine;

impl MetalEngine {
    pub fn api() -> Option<String>{
        Some("Metal".to_string())
    }

    pub fn supported() -> bool {
        Self::status()["renderer"] == "GPU"
    }

    pub fn status() -> Value {
        MTL_STATUS.get_or_init(||{
            render_thread::run(|_| {
                // if a context can be created successfully, collect info about the GPU
                let context = MetalContext::new().ok_or("GPU initialization failed".to_string())?;
                let device_name = format!("{} ({})", match context.device.location(){
                    MTLDeviceLocation::BuiltIn => "Integrated GPU",
                    MTLDeviceLocation::Slot => "Discrete GPU",
                    MTLDeviceLocation::External => "External GPU",
                    _ => "Other GPU"
                }, context.device.name());

                let status = json!({
                    "renderer": "GPU",
                    "api": "Metal",
                    "graphite": cfg!(feature = "graphite"),
                    "device": device_name,
                    "msaa": context.msaa,
                    "threads": rayon::current_num_threads(),
                });
                MTL_CONTEXT.set(Some(context));
                Ok(status)
            })
            .unwrap_or_else(|msg| json!({
                "renderer": "CPU",
                "api": "Metal",
                "device": "CPU-based renderer (Fallback)",
                "msaa": [0],
                "threads": rayon::current_num_threads(),
                "error": msg,
            }))
        }).clone()
    }

    pub fn with_context<T, F>(f:F) -> Result<T, String>
        where F:FnOnce(&mut MetalContext) -> Result<T, String>
    {
        match MetalEngine::supported() {
            false => Err("Metal API not supported".to_string()),
            true => MTL_CONTEXT.with_borrow_mut(|local_ctx|
                autoreleasepool(|_|
                    // lazily initialize this thread's context...
                    local_ctx
                        .take()
                        .or_else(|| MetalContext::new() )
                        .ok_or("Metal initialization failed".to_string())
                        .and_then(|ctx|{
                            f(local_ctx.insert(ctx))
                        })
                )
            )
        }
    }

    pub fn with_direct_context<F>(f:F)
        where F:FnOnce(Option<&mut DirectContext>)
    {
        Self::with_context(|ctx| Ok(f(Some(&mut ctx.context))) ).ok();
    }

    pub fn make_surface(image_info: &ImageInfo, opts:&ExportOptions, budgeted:bool) -> Result<Surface, String>{
        Self::with_context(|ctx| ctx.surface(image_info, opts, budgeted) )
    }

    // allow the render thread to check how long the context has gone unused
    pub fn context_is_idle() -> bool{
        MTL_CONTEXT.with_borrow(|cell|
            cell.as_ref().map(|engine| engine.last_use.elapsed() > MTL_CONTEXT_LIFESPAN).unwrap_or(false)
        )
    }

    pub fn retire(){
        MTL_CONTEXT.with_borrow_mut(|cell| { autoreleasepool(|_| { cell.take(); }); });
    }

    // called by the render thread between jobs to expire skia's internal caches
    pub fn purge_stale(){
        MTL_CONTEXT.with_borrow_mut(|cell| {
            if let Some(engine) = cell.as_mut(){
                autoreleasepool(|_|
                    engine.context.perform_deferred_cleanup(Duration::from_secs(1), None)
                );
            }
        });
    }

    // create a job-specific autorelease pool for each pass through the render-thread
    pub fn with_cleanup<T>(f: impl FnOnce() -> T) -> T {
        autoreleasepool(|_| f())
    }

    // read a gpu surface's pixels: Ganesh surfaces read directly, Graphite ones flush their recording first
    pub fn read_surface(surface:&mut Surface, info:&ImageInfo, pixels:&mut [u8], src:(i32, i32)) -> bool {
        #[cfg(feature = "graphite")]
        if let Ok(read) = Self::with_context(|ctx| Ok(ctx.graphite.read(surface, info, pixels, src))) {
            return read
        }
        surface.read_pixels(info, pixels, info.min_row_bytes(), src)
    }
}

// Chrome's GPU canvas draws with Skia Graphite (on Dawn over Metal); its path, arc and blur rasterization
// differs from Ganesh's, so with the `graphite` feature offscreen surfaces come from a Graphite recorder
#[cfg(feature = "graphite")]
pub struct GraphiteContext {
    context: skia_safe::gpu::graphite::Context,
    recorder: skia_safe::gpu::graphite::Recorder,
}

#[cfg(feature = "graphite")]
impl GraphiteContext {
    fn new(device:&ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        use skia_safe::gpu::graphite::mtl as gmtl;
        // Chrome's Dawn device lacks ShaderF16, so its shaders run at full precision; the fork's
        // Metal SkSL backend matches that (half -> float, f32 uniforms) when SK_MTL_F32 is set
        if std::env::var_os("SK_MTL_F32").is_none() {
            unsafe { std::env::set_var("SK_MTL_F32", "1") };
        }
        let queue = device.newCommandQueue()?;
        let backend = unsafe { gmtl::BackendContext::new(
            std::ptr::from_ref(device) as gmtl::Handle,
            Retained::as_ptr(&queue) as gmtl::Handle,
        ) };
        let mut context = gmtl::context_factory::make_metal(&backend, None)?;
        let recorder = context.make_recorder(None)?;
        Some(Self{ context, recorder })
    }

    fn surface(&mut self, image_info:&ImageInfo, opts:&ExportOptions) -> Option<Surface> {
        skia_safe::gpu::graphite::surfaces::render_target(
            &mut self.recorder, image_info, skia_safe::gpu::Mipmapped::No, Some(&opts.surface_props()), None
        )
    }

    fn read(&mut self, surface:&mut Surface, info:&ImageInfo, pixels:&mut [u8], src:(i32, i32)) -> bool {
        if let Some(mut recording) = self.recorder.snap() {
            self.context.insert_recording(&skia_safe::gpu::graphite::InsertRecordingInfo::new(&mut recording));
        }
        self.context.submit_and_wait();
        self.context.read_pixels(surface, info, pixels, info.min_row_bytes(), src)
    }
}

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    context: DirectContext,
    #[cfg(feature = "graphite")]
    graphite: GraphiteContext,
    msaa: Vec<usize>,
    last_use: Instant,
}

impl MetalContext{
    fn new() -> Option<Self>{
        autoreleasepool(|_| {
            MTLCreateSystemDefaultDevice().and_then(|device|{
                let last_use = Instant::now();
                let msaa:Vec<usize> = [0,2,4,8].into_iter().filter(|s|{
                    *s==0 || device.supportsTextureSampleCount(*s as _)
                }).collect();
                #[cfg(feature = "graphite")]
                let graphite = GraphiteContext::new(&device)?;
                make_direct_context(&device)
                    .map(|(_queue, context)| MetalContext{
                        device, context, msaa, last_use,
                        #[cfg(feature = "graphite")] graphite,
                    })
            })
        })
    }

    fn surface(&mut self, image_info: &ImageInfo, opts:&ExportOptions, #[allow(unused_variables)] budgeted:bool) -> Result<Surface, String> {
        self.last_use = Instant::now();
        #[cfg(feature = "graphite")]
        { // Graphite manages its own budget and sample count
            return self.graphite.surface(image_info, opts).ok_or(
                format!("Could not allocate new {}×{} bitmap", image_info.width(), image_info.height())
            )
        }
        surfaces::render_target(
            &mut self.context,
            if budgeted { Budgeted::Yes } else { Budgeted::No },
            image_info,
            Some(opts.msaa_from(&self.msaa)?),
            SurfaceOrigin::TopLeft,
            Some(&opts.surface_props()),
            false,
            None
        ).ok_or(
            format!("Could not allocate new {}×{} bitmap", image_info.width(), image_info.height())
        )
    }

}
