using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using ACadSharp;
using ACadSharp.Blocks;
using ACadSharp.Entities;
using ACadSharp.IO;
using ACadSharp.Objects;
using ACadSharp.Tables;
using CSMath;
using Viprs.Abi;
using Viprs.Sources;
using Viprs.Wire;

// The N1.4 view-context probe (libviprs/libviprs#1043).
//
//   vcprobe author <dir>              write the spike's own fixtures (vc_*.dwg)
//   vcprobe describe <dwg>            what ACadSharp's object model says: layers,
//                                     layouts, viewports, blocks, inserts
//   vcprobe shim <dwg>                what the libviprs-dep decoder emits, per view,
//                                     as the handle/flags/kind it puts on the wire
//   vcprobe rule <dwg>                the v1 visible-drawing rule applied to model
//                                     space, compared record for record with the
//                                     shim's output
//
// `rule` is the executable form of the rule in docs/spikes/view-context.md. It is
// written against ACadSharp's object model, never against the shim, so a
// disagreement is a finding about the shim and not a bug against itself.
namespace Viprs.Cad.Spike
{
	public static class Program
	{
		private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

		public static int Main(string[] args)
		{
			if (args.Length < 2)
			{
				Console.Error.WriteLine("usage: vcprobe author|describe|shim|rule <path>");
				return 2;
			}

			switch (args[0])
			{
				case "author":
					Author.WriteAll(args[1]);
					return 0;
				case "describe":
					Describe(args[1]);
					return 0;
				case "shim":
					Shim(args[1], false);
					return 0;
				case "rule":
					return Rule(args[1]);
				default:
					Console.Error.WriteLine("unknown mode " + args[0]);
					return 2;
			}
		}

		// ------------------------------------------------------------ helpers

		internal static CadDocument Read(string path)
		{
			using (DwgReader reader = new DwgReader(path))
			{
				return reader.Read();
			}
		}

		internal static bool IsFrozen(Layer l)
		{
			return (l.Flags & LayerFlags.Frozen) != 0;
		}

		internal static bool LayerVisible(Layer l)
		{
			return l == null || (l.IsOn && !IsFrozen(l));
		}

		internal static string F(double d)
		{
			string s = d.ToString("0.######", Inv);
			return s == "-0" ? "0" : s;
		}

		internal static string P(XYZ p)
		{
			return "(" + F(p.X) + "," + F(p.Y) + "," + F(p.Z) + ")";
		}

		internal static string H(ulong h)
		{
			return h.ToString("X", Inv);
		}

		internal static bool IsXref(BlockRecord b)
		{
			if (b == null || b.BlockEntity == null)
			{
				return false;
			}

			BlockTypeFlags f = b.BlockEntity.Flags;
			return (f & BlockTypeFlags.XRef) != 0 || (f & BlockTypeFlags.XRefOverlay) != 0 || b.IsUnloaded;
		}

		// ----------------------------------------------------------- describe

		private static void Describe(string path)
		{
			CadDocument doc = Read(path);
			Console.WriteLine("file " + Path.GetFileName(path) + " version " + doc.Header.Version);

			Console.WriteLine("layers " + doc.Layers.Count());
			foreach (Layer l in doc.Layers.OrderBy(x => x.Name, StringComparer.Ordinal))
			{
				Console.WriteLine(
					"  layer name=" + l.Name
					+ " on=" + l.IsOn
					+ " frozen=" + IsFrozen(l)
					+ " frozenNewVp=" + ((l.Flags & LayerFlags.FrozenNewViewports) != 0)
					+ " locked=" + ((l.Flags & LayerFlags.Locked) != 0)
					+ " plot=" + l.PlotFlag
					+ " xrefDep=" + ((l.Flags & LayerFlags.XrefDependent) != 0)
				);
			}

			Console.WriteLine("layouts " + doc.Layouts.Count());
			foreach (Layout lay in doc.Layouts.OrderBy(x => x.TabOrder))
			{
				BlockRecord blk = lay.AssociatedBlock;
				Console.WriteLine(
					"  layout name=" + lay.Name
					+ " paper=" + lay.IsPaperSpace
					+ " block=" + (blk == null ? "-" : blk.Name)
					+ " entities=" + (blk == null ? 0 : blk.Entities.Count)
					+ " ext=" + P(lay.MinExtents) + ".." + P(lay.MaxExtents)
				);
				if (blk == null)
				{
					continue;
				}

				foreach (Viewport vp in blk.Entities.OfType<Viewport>())
				{
					Console.WriteLine(
						"    viewport handle=" + H(vp.Handle)
						+ " id=" + vp.Id
						+ " layer=" + (vp.Layer == null ? "-" : vp.Layer.Name)
						+ " center=" + P(vp.Center)
						+ " w=" + F(vp.Width) + " h=" + F(vp.Height)
						+ " viewCenter=(" + F(vp.ViewCenter.X) + "," + F(vp.ViewCenter.Y) + ")"
						+ " viewHeight=" + F(vp.ViewHeight)
						+ " viewDir=" + P(vp.ViewDirection)
						+ " twist=" + F(vp.TwistAngle)
						+ " scale=" + F(vp.Height == 0 || vp.ViewHeight == 0 ? 0 : vp.Height / vp.ViewHeight)
						+ " status=" + (int)vp.Status
						+ " nonRect=" + ((vp.Status & ViewportStatusFlags.NonRectangularClipping) != 0)
						+ " off=" + ((vp.Status & ViewportStatusFlags.ViewportOff) != 0)
						+ " representsPaper=" + vp.RepresentsPaper
						+ " boundary=" + (vp.Boundary == null ? "-" : vp.Boundary.ObjectName)
						+ " frozenInVp=[" + string.Join(",", vp.FrozenLayers.Select(x => x.Name)) + "]"
					);
				}
			}

			Console.WriteLine("blocks " + doc.BlockRecords.Count());
			foreach (BlockRecord b in doc.BlockRecords.OrderBy(x => x.Name, StringComparer.Ordinal))
			{
				Dictionary<string, int> kinds = new Dictionary<string, int>();
				foreach (Entity e in b.Entities)
				{
					string k = e.ObjectName + "@" + (e.Layer == null ? "-" : e.Layer.Name);
					kinds[k] = kinds.TryGetValue(k, out int c) ? c + 1 : 1;
				}

				Console.WriteLine(
					"  block name=" + b.Name
					+ " flags=" + (b.BlockEntity == null ? "-" : b.BlockEntity.Flags.ToString())
					+ " anonymous=" + b.IsAnonymous
					+ " dynamic=" + b.IsDynamic
					+ " source=" + (SourceName(b))
					+ " xref=" + IsXref(b)
					+ " xrefPath=" + (b.BlockEntity == null ? "" : b.BlockEntity.XRefPath)
					+ " entities=" + b.Entities.Count
					+ " [" + string.Join(" ", kinds.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => kv.Key + "x" + kv.Value)) + "]"
				);
			}

			Console.WriteLine("inserts");
			foreach (BlockRecord owner in doc.BlockRecords.OrderBy(x => x.Name, StringComparer.Ordinal))
			{
				foreach (Insert ins in owner.Entities.OfType<Insert>())
				{
					Console.WriteLine(
						"  insert owner=" + owner.Name
						+ " handle=" + H(ins.Handle)
						+ " layer=" + (ins.Layer == null ? "-" : ins.Layer.Name)
						+ " block=" + (ins.Block == null ? "-" : ins.Block.Name)
						+ " at=" + P(ins.InsertPoint)
						+ " scale=(" + F(ins.XScale) + "," + F(ins.YScale) + "," + F(ins.ZScale) + ")"
						+ " rot=" + F(ins.Rotation)
						+ " grid=" + ins.RowCount + "x" + ins.ColumnCount
						+ " attribs=" + ins.Attributes.Count
					);
				}
			}
		}

		private static string SourceName(BlockRecord b)
		{
			try
			{
				BlockRecord s = b.Source;
				return s == null ? "-" : s.Name;
			}
			catch (Exception)
			{
				return "?";
			}
		}

		// --------------------------------------------------------------- shim

		private sealed class Emitted
		{
			public int View;
			public string ViewName;
			public ushort Type;
			public ulong Handle;
			public uint Flags;
			public double[] Values;
			public uint Code;
			public string Text;
		}

		private static List<Emitted> RunShim(string path, out List<string> viewLines)
		{
			List<Emitted> all = new List<Emitted>();
			viewLines = new List<string>();
			ResolvedLimits limits = new ResolvedLimits();
			using (IDocumentSource source = SourceFactory.OpenPath(path, limits))
			{
				for (int i = 0; i < source.ViewCount; i++)
				{
					SourceView v;
					source.TryGetView(i, out v);
					viewLines.Add(
						"view " + i + " name=" + v.Name + " kind=" + (v.Kind == 0 ? "model" : v.Kind == 1 ? "layout" : "unknown")
						+ " items=" + v.ItemCount
					);
					foreach (Primitive p in source.EnumerateView(i, null))
					{
						all.Add(
							new Emitted
							{
								View = i,
								ViewName = v.Name,
								Type = p.Type,
								Handle = p.ItemHandle,
								Flags = p.Flags,
								Values = p.Values,
								Code = p.Counts != null && p.Counts.Length > 0 ? p.Counts[0] : 0,
								Text = p.Text,
							}
						);
					}
				}
			}

			return all;
		}

		private static string KindName(ushort type)
		{
			switch (type)
			{
				case WireFormat.TypeLine: return "Line";
				case WireFormat.TypePolyline: return "Polyline";
				case WireFormat.TypeArc: return "Arc";
				case WireFormat.TypeCircle: return "Circle";
				case WireFormat.TypeEllipse: return "Ellipse";
				case WireFormat.TypeSpline: return "Spline";
				case WireFormat.TypePolygon: return "Polygon";
				case WireFormat.TypeText: return "Text";
				case WireFormat.TypeWarning: return "Warning";
				default: return "type" + type;
			}
		}

		private static void Shim(string path, bool quiet)
		{
			List<string> viewLines;
			List<Emitted> all = RunShim(path, out viewLines);
			foreach (string l in viewLines)
			{
				Console.WriteLine(l);
			}

			foreach (Emitted e in all)
			{
				if (e.Type == WireFormat.TypeWarning)
				{
					if (e.Code == 101)
					{
						continue; // reader notifications head every view; counted below
					}

					Console.WriteLine("  v" + e.View + " Warning code=" + e.Code + " handle=" + H(e.Handle) + " " + e.Text);
					continue;
				}

				if (!WireFormat.IsGeometry(e.Type))
				{
					continue;
				}

				string pts = e.Values == null || e.Values.Length == 0
					? ""
					: " v=[" + string.Join(",", e.Values.Take(6).Select(F)) + "]";
				Console.WriteLine("  v" + e.View + " " + KindName(e.Type) + " handle=" + H(e.Handle) + " flags=" + e.Flags + pts);
			}

			Console.WriteLine("reader_notifications " + all.Count(e => e.Type == WireFormat.TypeWarning && e.Code == 101) / Math.Max(1, viewLines.Count) + " per view");
		}

		// --------------------------------------------------------------- rule

		private sealed class Leaf
		{
			public ulong Handle;
			public string Kind;
			public string Layer;
		}

		// V1 rule, model space only. A leaf is an entity that is not an INSERT.
		//
		//   V1-1  only the model-space block contributes.
		//   V1-2  a leaf's effective layer is its own, except that a leaf on
		//         layer "0" inside an insertion takes the layer of the nearest
		//         enclosing INSERT.
		//   V1-3  an INSERT on an off or frozen layer contributes nothing,
		//         whatever its block holds, and a leaf on an off or frozen
		//         effective layer contributes nothing.
		//   V1-4  an INSERT of an xref contributes nothing and is reported.
		//   V1-5  everything else contributes once per instance, so one block
		//         inserted twice contributes its leaves twice.
		private static void ApplyRule(
			IEnumerable<Entity> entities,
			Layer insertLayer,
			int depth,
			List<Leaf> visible,
			List<string> external,
			List<string> hidden
		)
		{
			foreach (Entity e in entities)
			{
				Layer own = e.Layer;
				Layer effective = (insertLayer != null && own != null && own.Name == "0") ? insertLayer : own;
				if (!LayerVisible(effective))
				{
					hidden.Add(e.ObjectName + " " + H(e.Handle) + " on " + (effective == null ? "-" : effective.Name));
					continue;
				}

				Insert ins = e as Insert;
				if (ins != null)
				{
					if (ins.Block == null || IsXref(ins.Block))
					{
						external.Add("INSERT " + H(ins.Handle) + " -> " + (ins.Block == null ? "(no block)" : ins.Block.Name));
						continue;
					}

					int copies = Math.Max(1, (int)ins.RowCount) * Math.Max(1, (int)ins.ColumnCount);
					for (int c = 0; c < copies; c++)
					{
						ApplyRule(ins.Block.Entities, effective, depth + 1, visible, external, hidden);
					}

					continue;
				}

				// A DIMENSION draws its anonymous geometry block, so for the rule it is
				// an insertion of that block on the dimension's own layer.
				Dimension dim = e as Dimension;
				if (dim != null && dim.Block != null)
				{
					ApplyRule(dim.Block.Entities, effective, depth + 1, visible, external, hidden);
					continue;
				}

				visible.Add(new Leaf { Handle = e.Handle, Kind = e.ObjectName, Layer = effective == null ? "-" : effective.Name });
			}
		}

		private static int Rule(string path)
		{
			CadDocument doc = Read(path);
			List<Leaf> visible = new List<Leaf>();
			List<string> external = new List<string>();
			List<string> hidden = new List<string>();
			ApplyRule(doc.ModelSpace.Entities, null, 0, visible, external, hidden);

			List<string> viewLines;
			List<Emitted> shim = RunShim(path, out viewLines);
			List<Emitted> model = shim.Where(e => e.View == 0 && WireFormat.IsGeometry(e.Type)).ToList();

			// The comparison is made on one-record kinds only. A DIMENSION, HATCH,
			// MLINE or MESH fans out into several records under one handle, and a
			// POINT or XLINE is refused, so counting them would report the shim's
			// entity coverage rather than its layer handling. They are listed
			// separately by `shim`.
			HashSet<string> simple = new HashSet<string> { "LINE", "ARC", "CIRCLE", "ELLIPSE", "LWPOLYLINE", "SPLINE" };
			visible = visible.Where(l => simple.Contains(l.Kind)).ToList();

			// Multiset of handles, so one block instanced twice is two.
			Dictionary<ulong, int> want = new Dictionary<ulong, int>();
			foreach (Leaf l in visible)
			{
				want[l.Handle] = want.TryGetValue(l.Handle, out int c) ? c + 1 : 1;
			}

			Dictionary<ulong, string> kindOf = new Dictionary<ulong, string>();
			foreach (BlockRecord b0 in doc.BlockRecords)
			{
				foreach (Entity e0 in b0.Entities)
				{
					kindOf[e0.Handle] = e0.ObjectName;
				}
			}

			model = model.Where(e => kindOf.TryGetValue(e.Handle, out string k) && simple.Contains(k)).ToList();
			Dictionary<ulong, int> got = new Dictionary<ulong, int>();
			foreach (Emitted e in model)
			{
				got[e.Handle] = got.TryGetValue(e.Handle, out int c) ? c + 1 : 1;
			}

			// Handle -> (kind, layer) for naming, over every block.
			Dictionary<ulong, string> names = new Dictionary<ulong, string>();
			foreach (BlockRecord b in doc.BlockRecords)
			{
				foreach (Entity e in b.Entities)
				{
					names[e.Handle] = e.ObjectName + " on " + (e.Layer == null ? "-" : e.Layer.Name) + " in " + b.Name;
				}
			}

			int leaks = 0;
			int lost = 0;
			List<string> lines = new List<string>();
			foreach (ulong h in want.Keys.Union(got.Keys).OrderBy(x => x))
			{
				int w = want.TryGetValue(h, out int a) ? a : 0;
				int g = got.TryGetValue(h, out int b2) ? b2 : 0;
				if (w == g)
				{
					continue;
				}

				string what = names.TryGetValue(h, out string n) ? n : "?";
				if (g > w)
				{
					leaks += g - w;
					lines.Add("  EMITTED-BUT-HIDDEN handle=" + H(h) + " rule=" + w + " shim=" + g + " (" + what + ")");
				}
				else
				{
					lost += w - g;
					lines.Add("  VISIBLE-BUT-NOT-EMITTED handle=" + H(h) + " rule=" + w + " shim=" + g + " (" + what + ")");
				}
			}

			Console.WriteLine("file " + Path.GetFileName(path));
			Console.WriteLine("rule_visible_leaves " + visible.Count);
			Console.WriteLine("rule_hidden_entities " + hidden.Count);
			Console.WriteLine("rule_external " + external.Count);
			foreach (string x in external)
			{
				Console.WriteLine("  external " + x);
			}

			Console.WriteLine("shim_model_space_geometry_records " + model.Count);
			Console.WriteLine("shim_emitted_but_hidden_by_rule " + leaks);
			Console.WriteLine("shim_visible_by_rule_but_not_emitted " + lost);
			foreach (string l in lines.Take(40))
			{
				Console.WriteLine(l);
			}

			if (lines.Count > 40)
			{
				Console.WriteLine("  ... " + (lines.Count - 40) + " more");
			}

			foreach (Emitted e in shim.Where(e => e.View == 0 && e.Type == WireFormat.TypeWarning && e.Code != 101))
			{
				Console.WriteLine("  shim warning code=" + e.Code + " handle=" + H(e.Handle) + " " + e.Text);
			}

			return 0;
		}
	}
}
