//! A drawing served from memory for the #1713 plot tests: a model of `model` short lines on a grid
//! 1000 wide, and `sheets` paper layouts of 420 by 297 mm that each show a window of it through one
//! viewport (the last sheet through `1 + extra`). The window is `view_h` model units tall; with `only`
//! set the model holds just the lines that touch the window, which is what a plot must draw anyway.

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceColour, SourceFact,
    SourceLineweight, ViewportClip, ViewportEntity,
};

pub struct Synthetic {
    sheets: u32,
    model: u32,
    view_h: f64,
    only: bool,
    extra: u32,
}

/// The bytes that open as the drawing described in the module docs.
pub fn drawing(sheets: u32, model: u32, view_h: u32, only: bool, extra: u32) -> Vec<u8> {
    format!(
        "MOCK-1713-{sheets}-{model}-{view_h}-{}-{extra}",
        u8::from(only)
    )
    .into_bytes()
}

/// Every seventh line is long, so some cross the window's edge.
fn ends(i: u32) -> ([f64; 2], [f64; 2]) {
    let (x, y) = (f64::from(i % 1000), f64::from(i / 1000));
    let len = if i % 7 == 0 { 60.5 } else { 0.5 };
    ([x, y], [x + len, y + 0.25])
}

impl Synthetic {
    fn window(&self) -> [f64; 4] {
        [0.0, 0.0, self.view_h * 400.0 / 280.0, self.view_h]
    }

    fn viewport(&self, sheet: u32, k: u32) -> ViewportEntity {
        let (center, size) = (
            [210.0, 148.5],
            if k == 0 {
                [420.0, 297.0]
            } else {
                [400.0, 280.0]
            },
        );
        let (hw, hh) = (size[0] / 2.0, size[1] / 2.0);
        let w = self.window();
        ViewportEntity {
            handle: ItemHandle::new(0x10_0000_0000 + u64::from(sheet) * 64 + u64::from(k)),
            layout: LayoutId(sheet),
            id: i16::try_from(k + 1).unwrap_or(i16::MAX),
            visible: true,
            clip: ViewportClip::Rectangle([
                [center[0] - hw, center[1] - hh],
                [center[0] + hw, center[1] - hh],
                [center[0] + hw, center[1] + hh],
                [center[0] - hw, center[1] + hh],
            ]),
            frozen_layers: Vec::new(),
            layer: "VP".into(),
            center,
            width: size[0],
            height: size[1],
            view_center: if k == 0 {
                center
            } else {
                [w[2] / 2.0, w[3] / 2.0]
            },
            view_height: if k == 0 { 297.0 } else { self.view_h },
            view_target: [0.0, 0.0, 0.0],
            view_direction: [0.0, 0.0, 1.0],
            twist: 0.0,
            perspective: false,
            hide_plot: false,
            non_rectangular_clipping: false,
        }
    }

    fn per_sheet(&self, sheet: u32) -> u32 {
        if sheet == self.sheets {
            1 + self.extra
        } else {
            1
        }
    }
}

struct Model {
    next: u32,
    end: u32,
    only: Option<[f64; 4]>,
}

fn leaf(i: u32, handle: u64, class: &str, layer: &str, geometry: RawGeometry) -> RawEntity {
    RawEntity {
        style: EntityStyle {
            colour: SourceFact::Present(SourceColour::Aci(7)),
            lineweight: SourceFact::Present(SourceLineweight::Hundredths(25)),
            ..EntityStyle::absent()
        },
        geometry,
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle + u64::from(i))),
            class_name: class.into(),
            block_path: InstancePath::root(),
            layer: layer.into(),
        },
    }
}

impl EntityCursor for Model {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        while self.next < self.end {
            let i = self.next;
            self.next += 1;
            let (a, b) = ends(i);
            if let Some(w) = self.only
                && !(a[0].min(b[0]) <= w[2]
                    && a[0].max(b[0]) >= w[0]
                    && a[1].min(b[1]) <= w[3]
                    && a[1].max(b[1]) >= w[1])
            {
                continue;
            }
            return Ok(Some(leaf(
                i,
                1,
                "LINE",
                "0",
                RawGeometry::Line {
                    start: [a[0], a[1], 0.0],
                    end: [b[0], b[1], 0.0],
                },
            )));
        }
        Ok(None)
    }
}

struct Paper(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Paper {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

impl DwgProvider for Synthetic {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(matches!(source, CadSource::Bytes(b) if b.starts_with(b"MOCK-1713-")))
    }
    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let bytes = match source {
            CadSource::Bytes(b) => b.to_vec(),
            _ => Vec::new(),
        };
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let mut parts = text.strip_prefix("MOCK-1713-").unwrap_or("").split('-');
        let mut number = || parts.next().and_then(|n| n.parse::<u32>().ok());
        let (Some(sheets), Some(model), Some(view_h), Some(only), Some(extra)) =
            (number(), number(), number(), number(), number())
        else {
            return Err(ProviderError::Unrecognised {
                provider: "mock-1713",
                detail: "not a synthetic drawing".into(),
            });
        };
        Ok(Self {
            sheets,
            model,
            view_h: f64::from(view_h),
            only: only == 1,
            extra,
        })
    }
    fn name(&self) -> &'static str {
        "mock-1713"
    }
    fn caps(&self) -> &ProviderCaps {
        static CAPS: std::sync::OnceLock<ProviderCaps> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| {
            Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
                c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
            })
        })
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        let mut v = vec![LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        }];
        for i in 1..=self.sheets {
            v.push(LayoutState {
                id: LayoutId(i),
                name: format!("S{i:03}"),
                kind: LayoutKind::Paper,
                block_name: format!("*Paper_Space{i}"),
            });
        }
        Ok(v)
    }
    fn layout_plot(&self, layout: LayoutId) -> Result<SourceFact<LayoutPlot>, ProviderError> {
        if layout.0 == 0 || layout.0 > self.sheets {
            return Err(ProviderError::NoSuchScope(format!("layout {}", layout.0)));
        }
        Ok(SourceFact::Present(LayoutPlot {
            paper_size: "TEST_SHEET".into(),
            paper_width: 420.0,
            paper_height: 297.0,
            paper_units: 1,
            plot_type: 5,
            window: [0.0; 4],
            scale_numerator: 1.0,
            scale_denominator: 1.0,
            rotation: 0,
            margins: [0.0; 4],
            origin: [0.0; 2],
            paper_image_origin: [0.0; 2],
            limits: [0.0; 4],
            style_sheet: "monochrome.ctb".into(),
            print_lineweights: true,
            scale_lineweights: false,
            plot_plot_styles: true,
            plot_viewport_borders: false,
            draw_viewports_first: false,
        }))
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(["0", "VP"]
            .iter()
            .map(|n| LayerState {
                name: (*n).into(),
                ..LayerState::default()
            })
            .collect())
    }
    fn viewports(&self) -> Result<Vec<ViewportEntity>, ProviderError> {
        Ok((1..=self.sheets)
            .flat_map(|s| (0..=self.per_sheet(s)).map(move |k| (s, k)))
            .map(|(s, k)| self.viewport(s, k))
            .collect())
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        match scope {
            EntityScope::Layout(LayoutId(0)) => Ok(Box::new(Model {
                next: 0,
                end: self.model,
                only: self.only.then(|| self.window()),
            })),
            EntityScope::Layout(LayoutId(n)) if *n <= self.sheets => {
                let vps: Vec<RawEntity> = (0..=self.per_sheet(*n))
                    .map(|k| {
                        leaf(
                            0,
                            0x10_0000_0000 + u64::from(*n) * 64 + u64::from(k),
                            "VIEWPORT",
                            "VP",
                            RawGeometry::Unmapped {
                                reason: "a viewport is not geometry".into(),
                            },
                        )
                    })
                    .collect();
                Ok(Box::new(Paper(vps.into_iter())))
            }
            other => Err(ProviderError::NoSuchScope(format!("{other:?}"))),
        }
    }
}
