using System;
using System.IO;
using ACadSharp;
using ACadSharp.Blocks;
using ACadSharp.Entities;
using ACadSharp.IO;
using ACadSharp.Tables;
using CSMath;

// The spike's own fixtures, written with ACadSharp's DwgWriter at the pinned
// version. They are NOT known-truth fixtures in the pvec-gold-fixtures sense:
// the writer and the reader are the same library family, so a read-back proves
// the file says what I intended and nothing about what AutoCAD would draw. The
// document calls them "spike-authored" for that reason, and the real
// AutoCAD-authored drawings carry the cases the writer cannot honestly fake.
namespace Viprs.Cad.Spike
{
	internal static class Author
	{
		public static void WriteAll(string dir)
		{
			Directory.CreateDirectory(dir);
			Layers(Path.Combine(dir, "vc_layers.dwg"));
			Nested(Path.Combine(dir, "vc_nested.dwg"));
			Paper(Path.Combine(dir, "vc_paper.dwg"));
			Anonymous(Path.Combine(dir, "vc_anonymous.dwg"));
		}

		private static Layer AddLayer(CadDocument doc, string name, bool on = true, bool frozen = false)
		{
			Layer l = new Layer(name) { IsOn = on };
			if (frozen)
			{
				l.Flags |= LayerFlags.Frozen;
			}

			doc.Layers.Add(l);
			return l;
		}

		private static Line Ln(double x0, double y0, double x1, double y1, Layer layer)
		{
			return new Line(new XYZ(x0, y0, 0), new XYZ(x1, y1, 0)) { Layer = layer };
		}

		// Layer state in model space, and an INSERT of one block on each state.
		//   m1 on VC_ON, m2 on VC_OFF, m3 on VC_FROZEN, m4 on layer "0" (off)
		//   VC_BLK: b1 on "0", b2 on VC_ON, b3 on VC_OFF
		//   inserted on VC_ON (x=20), VC_OFF (x=30), VC_FROZEN (x=40)
		private static void Layers(string path)
		{
			CadDocument doc = new CadDocument();
			Layer on = AddLayer(doc, "VC_ON");
			Layer off = AddLayer(doc, "VC_OFF", on: false);
			Layer frozen = AddLayer(doc, "VC_FROZEN", frozen: true);
			Layer zero = doc.Layers["0"];
			zero.IsOn = false;

			doc.Entities.Add(Ln(0, 0, 10, 0, on));
			doc.Entities.Add(Ln(0, 2, 10, 2, off));
			doc.Entities.Add(Ln(0, 4, 10, 4, frozen));
			doc.Entities.Add(Ln(0, 6, 10, 6, zero));

			BlockRecord blk = new BlockRecord("VC_BLK");
			blk.Entities.Add(Ln(0, 0, 1, 0, zero));
			blk.Entities.Add(Ln(0, 1, 1, 1, on));
			blk.Entities.Add(Ln(0, 2, 1, 2, off));
			doc.BlockRecords.Add(blk);

			doc.Entities.Add(new Insert(blk) { InsertPoint = new XYZ(20, 0, 0), Layer = on });
			doc.Entities.Add(new Insert(blk) { InsertPoint = new XYZ(30, 0, 0), Layer = off });
			doc.Entities.Add(new Insert(blk) { InsertPoint = new XYZ(40, 0, 0), Layer = frozen });
			DwgWriter.Write(path, doc);
		}

		// One block twice at different scales, and a three-deep chain.
		//   INNER   line (0,0)-(1,0)
		//   MID     line (0,0)-(0,1); INSERT INNER at (1,0) rot 90deg scale 2
		//   OUTER   INSERT MID at (3,0) rot 90deg scale 1
		//   model   INNER at (10,0) s1; INNER at (20,0) s3; OUTER at (100,100) s2
		private static void Nested(string path)
		{
			CadDocument doc = new CadDocument();
			Layer on = AddLayer(doc, "VC_ON");

			BlockRecord inner = new BlockRecord("VC_INNER");
			inner.Entities.Add(Ln(0, 0, 1, 0, on));
			doc.BlockRecords.Add(inner);

			BlockRecord mid = new BlockRecord("VC_MID");
			mid.Entities.Add(Ln(0, 0, 0, 1, on));
			mid.Entities.Add(new Insert(inner) { InsertPoint = new XYZ(1, 0, 0), Rotation = Math.PI / 2, XScale = 2, YScale = 2, ZScale = 2, Layer = on });
			doc.BlockRecords.Add(mid);

			BlockRecord outer = new BlockRecord("VC_OUTER");
			outer.Entities.Add(new Insert(mid) { InsertPoint = new XYZ(3, 0, 0), Rotation = Math.PI / 2, Layer = on });
			doc.BlockRecords.Add(outer);

			doc.Entities.Add(new Insert(inner) { InsertPoint = new XYZ(10, 0, 0), Layer = on });
			doc.Entities.Add(new Insert(inner) { InsertPoint = new XYZ(20, 0, 0), XScale = 3, YScale = 3, ZScale = 3, Layer = on });
			doc.Entities.Add(new Insert(outer) { InsertPoint = new XYZ(100, 100, 0), XScale = 2, YScale = 2, ZScale = 2, Layer = on });
			DwgWriter.Write(path, doc);
		}

		// Paper space with four viewports.
		//   model   A on VC_ON (0,0)-(100,0); B on VC_VPF (0,10)-(100,10); circle C on VC_ON at (50,50) r20
		//   paper   title line on VC_ON
		//   vp1     the paper-image viewport (id 1), which ACadSharp's default document already carries
		//   vp2     rectangle 100x80 at (80,100), ViewCenter (50,5), ViewHeight 80 (scale 1)
		//   vp3     rectangle 100x80 at (200,100), ViewCenter (50,5), ViewHeight 160 (scale 0.5),
		//           twist 30 degrees, VC_VPF frozen in this viewport only
		//   vp4     40x40 at (80,30), non-rectangular clip, boundary a circle r20 in paper space
		private static void Paper(string path)
		{
			CadDocument doc = new CadDocument();
			Layer on = AddLayer(doc, "VC_ON");
			Layer vpf = AddLayer(doc, "VC_VPF");

			doc.Entities.Add(Ln(0, 0, 100, 0, on));
			doc.Entities.Add(Ln(0, 10, 100, 10, vpf));
			doc.Entities.Add(new Circle { Center = new XYZ(50, 50, 0), Radius = 20, Layer = on });

			BlockRecord ps = doc.PaperSpace;
			ps.Entities.Add(Ln(10, 10, 200, 10, on));
			ps.Entities.Add(new Viewport { Center = new XYZ(80, 100, 0), Width = 100, Height = 80, ViewCenter = new XY(50, 5), ViewHeight = 80 });

			Viewport v3 = new Viewport { Center = new XYZ(200, 100, 0), Width = 100, Height = 80, ViewCenter = new XY(50, 5), ViewHeight = 160, TwistAngle = Math.PI / 6 };
			v3.FrozenLayers.Add(vpf);
			ps.Entities.Add(v3);

			Circle clip = new Circle { Center = new XYZ(80, 30, 0), Radius = 20, Layer = on };
			ps.Entities.Add(clip);
			Viewport v4 = new Viewport { Center = new XYZ(80, 30, 0), Width = 40, Height = 40, ViewCenter = new XY(50, 50), ViewHeight = 40, Boundary = clip };
			v4.Status |= ViewportStatusFlags.NonRectangularClipping;
			ps.Entities.Add(v4);
			DwgWriter.Write(path, doc);
		}

		// An anonymous block and a named one, to see what name the decoder hands over.
		private static void Anonymous(string path)
		{
			CadDocument doc = new CadDocument();
			Layer wall = AddLayer(doc, "A-WALL");
			Layer col = AddLayer(doc, "S-COLS");

			BlockRecord anon = new BlockRecord("*U1");
			anon.Flags |= BlockTypeFlags.Anonymous;
			anon.Entities.Add(Ln(0, 0, 5, 0, wall));
			doc.BlockRecords.Add(anon);

			BlockRecord named = new BlockRecord("COL_400");
			named.Entities.Add(new Circle { Center = new XYZ(0, 0, 0), Radius = 0.2, Layer = col });
			doc.BlockRecords.Add(named);

			doc.Entities.Add(new Insert(anon) { InsertPoint = new XYZ(0, 0, 0), Layer = wall });
			doc.Entities.Add(new Insert(named) { InsertPoint = new XYZ(10, 10, 0), Layer = col });
			DwgWriter.Write(path, doc);
		}
	}
}
