//! A drawing served from memory for the #1704 plot tests: `sheets` paper layouts of 420 by 297
//! mm, each with `lines` lines made on demand, so the drawing itself weighs the same whatever the
//! sheet count and a growing peak can only be the plot's.

use std::fs;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceColour, SourceFact,
    SourceLineweight,
};

pub struct Synthetic {
    sheets: u32,
    lines: u32,
}

/// The bytes that open as a drawing of `sheets` sheets with `lines` lines each.
pub fn drawing(sheets: u32, lines: u32) -> Vec<u8> {
    format!("MOCK-1704-{sheets}-{lines}").into_bytes()
}

struct Lines {
    sheet: u32,
    next: u32,
    end: u32,
}

impl EntityCursor for Lines {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        if self.next >= self.end {
            return Ok(None);
        }
        let i = self.next;
        self.next += 1;
        // Spread over the paper, a different length each, so no two strokes are the same text.
        let x = 5.0 + f64::from(i % 400);
        let y = 5.0 + f64::from((i / 400) % 280) + f64::from(i % 7) * 0.013;
        Ok(Some(RawEntity {
            style: EntityStyle {
                colour: SourceFact::Present(SourceColour::Aci(7)),
                lineweight: SourceFact::Present(SourceLineweight::Hundredths(25)),
                ..EntityStyle::absent()
            },
            geometry: RawGeometry::Line {
                start: [x, y, 0.0],
                end: [x + 3.0 + f64::from(i % 5) * 0.37, y + 1.0, 0.0],
            },
            provenance: ProvenanceStub {
                handle: Some(ItemHandle::new(
                    u64::from(self.sheet) * 10_000_000 + u64::from(i) + 1,
                )),
                class_name: "LINE".into(),
                block_path: InstancePath::root(),
                layer: "0".into(),
            },
        }))
    }
}

impl DwgProvider for Synthetic {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(matches!(source, CadSource::Bytes(b) if b.starts_with(b"MOCK-1704-")))
    }
    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let bytes = match source {
            CadSource::Bytes(b) => b.to_vec(),
            CadSource::Path(p) => fs::read(p).map_err(|e| ProviderError::Open {
                provider: "mock-1704",
                detail: e.to_string(),
            })?,
            _ => Vec::new(),
        };
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let mut parts = text.strip_prefix("MOCK-1704-").unwrap_or("").split('-');
        let mut number = || parts.next().and_then(|n| n.parse::<u32>().ok());
        let (Some(sheets), Some(lines)) = (number(), number()) else {
            return Err(ProviderError::Unrecognised {
                provider: "mock-1704",
                detail: "not a synthetic drawing".into(),
            });
        };
        Ok(Self { sheets, lines })
    }
    fn name(&self) -> &'static str {
        "mock-1704"
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
            plot_viewport_borders: true,
            draw_viewports_first: false,
        }))
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(vec![LayerState {
            name: "0".into(),
            ..LayerState::default()
        }])
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        match scope {
            EntityScope::Layout(LayoutId(0)) => Ok(Box::new(Lines {
                sheet: 0,
                next: 0,
                end: 0,
            })),
            EntityScope::Layout(LayoutId(n)) if *n <= self.sheets => Ok(Box::new(Lines {
                sheet: *n,
                next: 0,
                end: self.lines,
            })),
            other => Err(ProviderError::NoSuchScope(format!("{other:?}"))),
        }
    }
}
