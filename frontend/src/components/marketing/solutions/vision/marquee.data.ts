import {
  Activity, Apple, BarChart3, Binary, Blocks, Bird, Bot, Boxes, Bug, Car,
  Cctv, CircleParking, CircuitBoard, ClipboardCheck, Cloud, Cog, Cpu, Database,
  FileText, Flame, FolderKanban, Gauge, Grid3x3, HardHat, Layers, LayoutGrid,
  Microscope, Package, PersonStanding, QrCode, RadioTower, Recycle, RefreshCw,
  Rocket, Scan, ScanLine, ScrollText, Server, ShieldCheck, ShoppingCart,
  Sparkles, SquareStack, Stethoscope, Tags, Target, Terminal, TestTube, Users,
  Waves, Webhook, Wheat, Zap,
} from "lucide-react";

/* =============================================================================
   Marquee content — /solutions/computer-vision only.

   Two tickers run off this file: the platform capabilities (what the product
   holds) and the applications (what people build with it). Both are plain data
   so the components stay presentational and a copy change never touches markup.

   Ground rule inherited from solutions.config.ts: every line here must resolve
   to something that ships today. That is why the deploy row names Raspberry Pi,
   UNO Q, PXE and TensorFlow Lite — the targets in HARDWARE_TARGETS — and not
   boards Petal Edge does not build for.
   ============================================================================= */

/** The five stages of the pipeline, in the order the platform runs them.
    A capability carries its stage, and the stage carries the colour — so the
    ticker's colour coding says where in the workflow each thing lives. */
export type StageId = "data" | "train" | "evaluate" | "deploy" | "operate";

export const STAGES: { id: StageId; label: string; accent: string }[] = [
  { id: "data", label: "Data", accent: "#6366f1" },
  { id: "train", label: "Train", accent: "#8b5cf6" },
  { id: "evaluate", label: "Evaluate", accent: "#06b6d4" },
  { id: "deploy", label: "Deploy", accent: "#22c55e" },
  { id: "operate", label: "Operate", accent: "#3b82f6" },
];

const STAGE_MAP = Object.fromEntries(STAGES.map((s) => [s.id, s])) as Record<
  StageId,
  (typeof STAGES)[number]
>;

export const stageOf = (id: StageId) => STAGE_MAP[id];

export type Capability = {
  title: string;
  stage: StageId;
  icon: React.ElementType;
};

export type Application = {
  title: string;
  desc: string;
  accent: string;
  icon: React.ElementType;
};

/* ─── Row 1 — everything before a model exists ─────────────────────────────── */
export const CAPABILITIES_DATA: Capability[] = [
  { title: "Dataset management", stage: "data", icon: Database },
  { title: "Image labeling", stage: "data", icon: Tags },
  { title: "AI-assisted labeling", stage: "data", icon: Sparkles },
  { title: "Bounding boxes", stage: "data", icon: Scan },
  { title: "Dataset versions", stage: "data", icon: SquareStack },
  { title: "Version control", stage: "data", icon: Layers },
  { title: "Projects", stage: "data", icon: FolderKanban },
  { title: "Users and access", stage: "data", icon: Users },
  { title: "Supported formats", stage: "data", icon: FileText },
  { title: "Industrial cameras", stage: "data", icon: Cctv },
];

/* ─── Row 2 — building the model and finding out whether it is any good ────── */
export const CAPABILITIES_MODEL: Capability[] = [
  { title: "Impulse design", stage: "train", icon: Blocks },
  { title: "Feature generation", stage: "train", icon: Waves },
  { title: "Vision Pro training", stage: "train", icon: Target },
  { title: "NanoVision", stage: "train", icon: Grid3x3 },
  { title: "Transfer learning", stage: "train", icon: Layers },
  { title: "GPU training", stage: "train", icon: Server },
  { title: "Cloud training", stage: "train", icon: Cloud },
  { title: "Evaluation", stage: "evaluate", icon: Gauge },
  { title: "Confusion matrix", stage: "evaluate", icon: LayoutGrid },
  { title: "Model testing", stage: "evaluate", icon: TestTube },
];

/* ─── Row 3 — getting it onto hardware and keeping it there ────────────────── */
export const CAPABILITIES_SHIP: Capability[] = [
  { title: "Model optimization", stage: "deploy", icon: Zap },
  { title: "Edge deployment", stage: "deploy", icon: Rocket },
  { title: "OTA updates", stage: "deploy", icon: RefreshCw },
  { title: "Fleet management", stage: "deploy", icon: RadioTower },
  { title: "Raspberry Pi", stage: "deploy", icon: CircuitBoard },
  { title: "Arduino UNO Q", stage: "deploy", icon: Cpu },
  { title: "PXE runtime", stage: "deploy", icon: Binary },
  { title: "Device monitoring", stage: "operate", icon: Activity },
  { title: "Inference logs", stage: "operate", icon: ScrollText },
  { title: "Analytics", stage: "operate", icon: BarChart3 },
  { title: "REST API", stage: "operate", icon: Webhook },
  { title: "Python SDK", stage: "operate", icon: Terminal },
];

/* ─── Applications — two rows, opposite directions ─────────────────────────── */
const IND = "#6366f1";
const VIO = "#8b5cf6";
const CYA = "#06b6d4";
const GRN = "#22c55e";
const BLU = "#3b82f6";

export const APPLICATIONS_A: Application[] = [
  { title: "Object detection", desc: "Put a box and a class on every instance in frame.", accent: VIO, icon: Target },
  { title: "People counting", desc: "Count how many people pass a fixed camera.", accent: IND, icon: Users },
  { title: "PPE detection", desc: "Check the required gear is worn before work starts.", accent: GRN, icon: ShieldCheck },
  { title: "Defect detection", desc: "Flag the parts that fail, and show where they fail.", accent: CYA, icon: Bug },
  { title: "Surface inspection", desc: "Grade finish and texture straight off the line.", accent: BLU, icon: Microscope },
  { title: "Barcode reading", desc: "Locate codes on moving stock so a reader can lock on.", accent: VIO, icon: QrCode },
  { title: "Text region OCR", desc: "Find the printed regions a reader needs to parse.", accent: IND, icon: ScanLine },
  { title: "Quality inspection", desc: "Sort output into the pass and fail classes you track.", accent: GRN, icon: ClipboardCheck },
  { title: "Retail analytics", desc: "See how shoppers move through a floor plan.", accent: CYA, icon: ShoppingCart },
  { title: "Vehicle detection", desc: "Pick out vehicles arriving at a yard or gate.", accent: BLU, icon: Car },
  { title: "Parking monitoring", desc: "Track which bays are free, bay by bay.", accent: VIO, icon: CircleParking },
  { title: "Waste sorting", desc: "Separate recyclable material on a sorting belt.", accent: GRN, icon: Recycle },
  { title: "Agriculture", desc: "Spot crop stress from a camera out in the field.", accent: IND, icon: Wheat },
];

export const APPLICATIONS_B: Application[] = [
  { title: "Wildlife monitoring", desc: "Identify species on trail cameras with no uplink.", accent: GRN, icon: Bird },
  { title: "Smart factory", desc: "Watch a cell and react when its state changes.", accent: IND, icon: Cog },
  { title: "Production counting", desc: "Count finished units without a mechanical trigger.", accent: VIO, icon: Boxes },
  { title: "Package detection", desc: "Find parcels on a belt and read their placement.", accent: CYA, icon: Package },
  { title: "Helmet detection", desc: "Confirm a helmet is on before the line runs.", accent: BLU, icon: HardHat },
  { title: "Fire and smoke", desc: "Catch smoke early in a fixed camera view.", accent: VIO, icon: Flame },
  { title: "License plates", desc: "Locate plates at a gate for a reader to parse.", accent: IND, icon: Scan },
  { title: "Posture checks", desc: "Detect the body positions that matter as their own classes.", accent: CYA, icon: PersonStanding },
  { title: "Fruit classification", desc: "Sort produce by ripeness and grade.", accent: GRN, icon: Apple },
  { title: "Medical imaging", desc: "Classify captured images in a controlled setup.", accent: BLU, icon: Stethoscope },
  { title: "Industrial automation", desc: "Give a controller a vision signal it can act on.", accent: VIO, icon: Terminal },
  { title: "Robot vision", desc: "Let a robot see the part before it reaches for it.", accent: IND, icon: Bot },
];
