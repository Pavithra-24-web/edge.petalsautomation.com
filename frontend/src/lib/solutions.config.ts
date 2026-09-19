/* =============================================================================
   Solutions — shared data model + content config
   Drives /solutions (hub) and /solutions/[slug] (9 pages) from one template.

   Ground rule: every claim here must resolve to code that ships today.
   See solutionspage.md §0 for the verification baseline and §3.1 for the
   copy constraints (no anomaly detection, no ESP32, no object_detection /
   yolov5 as product names, no accuracy figures).

   Icons are stored as *names* (see components/marketing/solutions/icons.tsx)
   so this module stays plain serializable data and can be imported from both
   server and client components.
   ============================================================================= */

export type SolutionGroup = "applications" | "industries" | "integrations";

export type WorkflowVariantId = "vision" | "sensor" | "integration";

export type HardwareTargetId =
  | "tflite"
  | "cpp"
  | "arduino"
  | "raspberry_pi"
  | "unoq"
  | "pxe";

/** Feature ids map 1:1 to the fixed pool in solutionspage.md §2.4. */
export type FeatureId =
  | "F1" | "F2" | "F3" | "F4" | "F5" | "F6" | "F7" | "F8" | "F9" | "F10"
  | "F11" | "F12" | "F13" | "F14" | "F15" | "F16" | "F17" | "F18" | "F19" | "F20";

/** Lucide icon names — resolved by solutions/icons.tsx. Keep in sync. */
export type IconName =
  | "Activity" | "BarChart3" | "Binary" | "BrainCircuit" | "Cable" | "Cherry"
  | "CircuitBoard" | "Cloud" | "CloudUpload" | "Code2" | "Cpu" | "Database"
  | "FileText" | "FlaskConical" | "Gauge" | "GitBranch" | "Layers"
  | "MousePointerClick" | "Package" | "PlayCircle" | "PlugZap" | "Radar"
  | "RadioTower" | "Rocket" | "Server" | "ShieldCheck" | "Sparkles" | "Wand2";

export type MediaAsset = {
  /** Path under /public. Omit the whole asset to render the section text-only. */
  src: string;
  alt: string;
  /** "image" renders <img>; "video" renders a muted, looping <video>. */
  kind: "image" | "video";
  poster?: string;
  caption?: string;
};

export type WorkflowStage = {
  name: string;
  desc: string;
  /**
   * Where the work happens, named the way a reader sees it in the product.
   * Never an API path, route or filename — this string is rendered on a public
   * page. The route each stage maps to is recorded in a comment beside it.
   */
  where: string;
  icon: IconName;
  /**
   * The stage's own capture, shown in the walkthrough carousel. Optional and
   * per-stage — no two stages share an asset, and nothing here is borrowed from
   * the homepage. Omit it and the slide renders its composed frame instead, at
   * the size the real capture will occupy, so adding one is a one-line change:
   *
   *   media: { kind: "image", src: "/workflow/label.png", alt: "…" }
   *   media: { kind: "video", src: "/workflow/train.mp4", poster: "…", alt: "…" }
   */
  media?: MediaAsset;
};

export type UseCase = {
  title: string;
  desc: string;
  /** Architecture actually used — must exist in model_builder.build_model. */
  model: string;
  /** Deployment target id, or a short phrase for non-deploy outcomes. */
  target: string;
};

export type Solution = {
  slug: string;
  group: SolutionGroup;
  name: string;
  /** One-line summary used on the hub card and in nav. */
  tagline: string;
  hero: {
    title: string;
    /** Trailing phrase rendered with the brand gradient. */
    titleAccent?: string;
    subtitle: string;
    /** 3 short proof chips. */
    chips: string[];
    /** Optional in-page anchor shown next to the fixed primary CTA. */
    secondary?: { label: string; href: string };
  };
  problem: {
    statement: string;
    audience: string[];
  };
  /** One or more workflow variants rendered in order. */
  workflow: WorkflowVariantId[];
  /** Exactly 6 ids from FEATURE_POOL. */
  features: FeatureId[];
  useCases: UseCase[];
  /**
   * "focused" — Application and Industry pages. Show only the targets that make
   * sense for that solution, then send the reader to Integrations for the rest.
   * These pages are not deployment documentation.
   * "full" — Integration pages. Show every supported target with its intended
   * usage; this is where the complete picture lives.
   */
  hardwareMode: "focused" | "full";
  hardware: HardwareTargetId[];
  /** Cross-links rendered under the deployment-options section. */
  related: { label: string; href: string }[];
  ctaHeadline: string;
  /**
   * Page metadata. Lives here rather than inline in the route so copy stays in
   * one file. `title` is the full <title>; `description` is the meta and OG
   * description and should read as a sentence, not a keyword list.
   */
  seo: { title: string; description: string };
  /** Assets land in Phase 3/4 — omitted keys render text-only. */
  media?: {
    hero?: MediaAsset;
    features?: MediaAsset;
    useCases?: MediaAsset;
    hardware?: MediaAsset;
  };
};

/* ─── Site metadata ─────────────────────────────────────────────────────────── */

/** Base for canonical and OG URLs. Override per environment; no trailing slash. */
export const SITE_URL = (
  process.env.NEXT_PUBLIC_SITE_URL || "https://petaledge.ai"
).replace(/\/$/, "");

export const SITE_NAME = "Petal Edge";

/* ─── Groups ────────────────────────────────────────────────────────────────── */

/**
 * The three groups answer three different questions and must never blur into
 * each other: Applications = what you build, Industries = where you apply it,
 * Integrations = how you deploy it. `kicker` is the one-line label reused in the
 * navbar dropdown and the mobile panel; `desc` is the longer hub-section lede.
 */
export const GROUPS: {
  id: SolutionGroup;
  label: string;
  kicker: string;
  desc: string;
  icon: IconName;
}[] = [
    {
      id: "applications",
      label: "Applications",
      kicker: "What you build",
      desc: "The kinds of model you can train here, and what each one is good at.",
      icon: "Sparkles",
    },
    {
      id: "industries",
      label: "Industries",
      kicker: "Where you apply it",
      desc: "Where those models end up, and what each setting demands of them.",
      icon: "Layers",
    },
    {
      id: "integrations",
      label: "Integrations",
      kicker: "How you deploy it",
      desc: "The targets a trained model can be built for, and what each one expects.",
      icon: "Package",
    },
  ];

/* ─── Fixed: why edge inference (identical on all 9 pages) ──────────────────── */

export const EDGE_INFERENCE_BULLETS: {
  title: string;
  desc: string;
  icon: IconName;
  accent: string;
}[] = [
    {
      icon: "Gauge",
      accent: "#8b5cf6",
      title: "Decisions happen locally",
      desc: "Inference runs on the device itself. There is no round trip to a server between a frame and a result.",
    },
    {
      icon: "Cloud",
      accent: "#3b82f6",
      title: "Keeps working offline",
      desc: "A deployed model runs without a connection. Devices send heartbeats and inference logs when they reconnect.",
    },
    {
      icon: "ShieldCheck",
      accent: "#22c55e",
      title: "Data stays where it is captured",
      desc: "Frames are processed on the device. Nothing has to leave the site for a prediction to be made.",
    },
  ];

/* ─── Fixed: feature pool (solutionspage.md §2.4) ───────────────────────────────
   Each entry carries a `// traces to:` comment naming the route or module that
   substantiates it. That trace is DELIBERATELY a comment and not a field: it is
   how a future edit stays checkable, and it must never reach the rendered page.
   Nothing in this table may print an API path, a route, a filename or a builder
   identifier.
   ---------------------------------------------------------------------------- */

export const FEATURE_POOL: Record<
  FeatureId,
  { title: string; desc: string; icon: IconName; accent: string }
> = {
  // traces to: /dashboard/data/import · /dashboard/data/csv-wizard
  F1: {
    title: "Dataset upload and import",
    desc: "Upload samples, import a folder tree, or map columns with the CSV wizard.",
    icon: "CloudUpload", accent: "#8b5cf6",
  },
  // traces to: samples.py import parsers · /dashboard/docs/supported-formats
  F2: {
    title: "Bring your existing annotations",
    desc: "Import COCO, YOLO, Pascal VOC, Open Images and Edge Impulse JSON without reformatting them first.",
    icon: "FileText", accent: "#6366f1",
  },
  // traces to: /dashboard/data/dataset — Konva editor, inline edit in grid
  F3: {
    title: "Bounding-box annotation editor",
    desc: "Draw, adjust and relabel boxes directly on the dataset grid.",
    icon: "MousePointerClick", accent: "#3b82f6",
  },
  // traces to: POST /ai-labeling/run · /apply · /reject (OWL-ViT zero-shot)
  F4: {
    title: "AI-assisted labeling",
    desc: "Describe what you are looking for in plain text and review the proposed boxes before they land.",
    icon: "Wand2", accent: "#06b6d4",
  },
  // traces to: GET /dsp/processing-blocks · GET /dsp/learning-blocks
  F5: {
    title: "Impulse designer",
    desc: "Chain a processing block and a learning block into a reproducible pipeline.",
    icon: "Layers", accent: "#14b8a6",
  },
  // traces to: POST /dsp/preview · POST /dsp/generate-features
  F6: {
    title: "Feature generation and preview",
    desc: "Preview a block's output on one sample, then generate features for the whole dataset.",
    icon: "Sparkles", accent: "#22c55e",
  },
  // traces to: POST /training/start · /training/retrain · /training/{job_id}/cancel
  F7: {
    title: "Training with live job logs",
    desc: "Watch a training job as it runs, cancel it, or retrain from an existing impulse.",
    icon: "Gauge", accent: "#8b5cf6",
  },
  // traces to: fomo_mobilenetv2_0_1 (v1/v2 via fomo_version) + yolo_pro builders
  F8: {
    title: "NanoVision and Vision Pro detection",
    desc: "Centroid detection with NanoVision v1 or v2, or box detection with Vision Pro.",
    icon: "Radar", accent: "#6366f1",
  },
  // traces to: mobilenet + transfer builders in model_builder.py
  F9: {
    title: "Image classification",
    desc: "Train MobileNetV2 from scratch or fine-tune with ImageNet transfer learning.",
    icon: "BrainCircuit", accent: "#3b82f6",
  },
  // traces to: dense/conv1d/conv2d/lstm builders + spectral, MFCC, spectrogram blocks
  F10: {
    title: "Sensor and audio models",
    desc: "Dense, 1D/2D convolutional and LSTM networks over spectral, MFCC and spectrogram features.",
    icon: "Activity", accent: "#06b6d4",
  },
  // traces to: GET /evaluation/job/{id}/confusion-matrix · /metrics
  F11: {
    title: "Evaluation you can read",
    desc: "Confusion matrix and per-class metrics for every completed training job.",
    icon: "BarChart3", accent: "#14b8a6",
  },
  // traces to: POST /model-testing/classify-all · GET /model-testing/model-versions
  F12: {
    title: "Model testing across versions",
    desc: "Run the test set through any model version and compare the results side by side.",
    icon: "FlaskConical", accent: "#22c55e",
  },
  // traces to: services/post_processing/{detection_postprocess,nms,tracker}.py
  F13: {
    title: "Detection post-processing",
    desc: "Confidence threshold, class filter, non-maximum suppression and IoU tracking, tuned in the UI.",
    icon: "GitBranch", accent: "#8b5cf6",
  },
  // traces to: POST /projects/{id}/post-processing-video + job video fetch
  F14: {
    title: "Video post-processing preview",
    desc: "Run a clip through the pipeline and watch the rendered result before you commit the settings.",
    icon: "PlayCircle", accent: "#6366f1",
  },
  // traces to: /dashboard/impulse/live-classification · POST /inference/predict
  F15: {
    title: "Live classification",
    desc: "Send a frame or a device stream through the current model and see the prediction immediately.",
    icon: "Cable", accent: "#3b82f6",
  },
  // traces to: GET /deployment/targets · POST /deployment/build
  F16: {
    title: "Deployment builds",
    desc: "Turn any trained model into a ready-to-run package for the target you pick.",
    icon: "Package", accent: "#06b6d4",
  },
  // traces to: devices.py — register · heartbeat · project inference-logs
  F17: {
    title: "Device fleet",
    desc: "Register devices with an API key, then track heartbeats and inference logs per project.",
    icon: "Server", accent: "#14b8a6",
  },
  // traces to: POST /devices/{pk}/request-update · /update-status · /compatible-deployment
  F18: {
    title: "Over-the-air model updates",
    desc: "Resolve the compatible deployment for a device and push the update without touching the hardware.",
    icon: "RadioTower", accent: "#22c55e",
  },
  // traces to: GET /projects/{id}/dataset-health
  F19: {
    title: "Dataset health checks",
    desc: "See class imbalance and dataset problems before you spend a training run on them.",
    icon: "ShieldCheck", accent: "#8b5cf6",
  },
  // traces to: POST /projects/{id}/export
  F20: {
    title: "Project export",
    desc: "Export the project so your data and models stay portable.",
    icon: "Database", accent: "#6366f1",
  },
};

/* ─── Fixed: workflow variants (solutionspage.md §2.3) ──────────────────────── */

export const WORKFLOW_VARIANTS: Record<
  WorkflowVariantId,
  { label: string; desc: string; stages: WorkflowStage[] }
> = {
  vision: {
    label: "Vision pipeline",
    desc: "From the first uploaded image to a monitored device, every stage is a screen you can open.",
    stages: [
      // /dashboard/data · /dashboard/data/import
      {
        name: "Collect data", desc: "Upload samples or import an existing folder tree.", where: "Data", icon: "CloudUpload",
        media: { kind: "image", src: "/collect_data.png", alt: "The data acquisition screen with a drop target for images and the list of uploaded samples beside it." },
      },
      // /dashboard/data/dataset
      {
        name: "Label", desc: "Draw boxes by hand, or start from AI-assisted suggestions.", where: "Dataset editor", icon: "MousePointerClick",
        media: { kind: "image", src: "/Label.png", alt: "The dataset editor with bounding boxes drawn over a sample and its class labels listed alongside." },
      },
      // /dashboard/impulse
      {
        name: "Design impulse", desc: "Pair the image processing block with a learning block.", where: "Impulse designer", icon: "Layers",
        // The file on disk has a space in its name; the URL keeps it encoded.
        media: { kind: "image", src: "/Design%20impulse.png", alt: "The impulse designer showing an input block wired to an image processing block and a learning block." },
      },
      // /dashboard/impulse/image/parameters · /image/generate-features
      {
        name: "Generate features", desc: "Process the dataset once and reuse the features.", where: "Feature generation", icon: "Sparkles",
        media: { kind: "image", src: "/Generate_features.png", alt: "The feature generation screen with the processed dataset plotted as a feature explorer scatter." },
      },
      // /dashboard/impulse/training
      {
        name: "Train", desc: "NanoVision, Vision Pro, MobileNetV2 or transfer learning.", where: "Training", icon: "Gauge",
        media: { kind: "image", src: "/Train.png", alt: "The training screen with the architecture settings on the left and live epoch metrics on the right." },
      },
      // /dashboard/impulse/evaluation
      {
        name: "Evaluate", desc: "Confusion matrix and per-class metrics for the run.", where: "Evaluation", icon: "BarChart3",
        media: { kind: "image", src: "/Evaluate.png", alt: "The evaluation screen with a confusion matrix above a per-class table of precision and recall." },
      },
      // /dashboard/impulse/model-testing
      {
        name: "Test the model", desc: "Score the test set and compare model versions.", where: "Model testing", icon: "FlaskConical",
        media: { kind: "image", src: "/Test_model.png", alt: "The model testing screen scoring the held-out test set, with per-sample results and a version comparison." },
      },
      // /dashboard/impulse/post-processing
      {
        name: "Post-process", desc: "Threshold, class filter, NMS and IoU tracking.", where: "Post-processing", icon: "GitBranch",
        media: { kind: "image", src: "/Postprocess.png", alt: "The post-processing screen with confidence threshold, class filter and NMS controls over a preview of the filtered detections." },
      },
      // /dashboard/impulse/deployment
      {
        name: "Deploy", desc: "Build a package for the target you actually run.", where: "Deployment", icon: "Package",
        media: { kind: "image", src: "/Deploy.png", alt: "The deployment screen with the selectable build targets and the trained model chosen for the package." },
      },
      // /dashboard/devices
      {
        name: "Run and monitor", desc: "Heartbeats, inference logs and over-the-air updates.", where: "Devices", icon: "Server",
        media: {
          kind: "video",
          src: "/Home_edit_1.web.mp4",
          poster: "/Home_edit_1_poster.jpg",
          alt: "A deployed model running on a live camera feed, labelling people and vehicles frame by frame with confidence scores.",
        },
      },
    ],
  },
  sensor: {
    label: "Sensor and audio pipeline",
    desc: "The same platform, driven by signals instead of frames.",
    stages: [
      // POST /ingestion/training/data — device SDK ingestion
      { name: "Ingest signals", desc: "Send samples straight from a device into the project.", where: "Device ingestion", icon: "RadioTower" },
      // /dashboard/data
      { name: "Label", desc: "Organize samples by class and split training from testing.", where: "Data", icon: "MousePointerClick" },
      // /dashboard/impulse
      { name: "Design impulse", desc: "Spectral analysis, MFCC, spectrogram, flatten or raw.", where: "Impulse designer", icon: "Layers" },
      // /dashboard/impulse/image/generate-features
      { name: "Generate features", desc: "Preview a block's output, then process the dataset.", where: "Feature generation", icon: "Sparkles" },
      // /dashboard/impulse/training
      { name: "Train", desc: "Dense, 1D/2D convolutional or LSTM networks.", where: "Training", icon: "Activity" },
      // /dashboard/impulse/evaluation
      { name: "Evaluate", desc: "Confusion matrix and per-class metrics.", where: "Evaluation", icon: "BarChart3" },
      // /dashboard/impulse/model-testing
      { name: "Test the model", desc: "Score the held-out test set across versions.", where: "Model testing", icon: "FlaskConical" },
      // /dashboard/impulse/deployment
      { name: "Deploy", desc: "Build the package for your target hardware.", where: "Deployment", icon: "Package" },
    ],
  },
  integration: {
    label: "Deployment path",
    desc: "What happens between a trained model and hardware running it in the field.",
    stages: [
      // /dashboard/impulse/deployment
      { name: "Build the package", desc: "Pick a target and build from any trained model.", where: "Deployment", icon: "Package" },
      // GET /deployment/{deployment_id} — artifact download
      { name: "Download", desc: "Collect the generated package for that target.", where: "Deployment", icon: "CloudUpload" },
      // device_client/install.sh + systemd unit
      { name: "Install on device", desc: "Run the device client installer and its service.", where: "Device client", icon: "Server" },
      // /dashboard/devices/connect
      { name: "Register", desc: "Claim the device with a project API key.", where: "Connect a device", icon: "ShieldCheck" },
      // /dashboard/impulse/live-classification
      { name: "Run inference", desc: "Stream results back while the model runs locally.", where: "Live classification", icon: "Cable" },
      // POST /devices/{pk}/request-update · /update-status
      { name: "Update over the air", desc: "Resolve the compatible build and push it out.", where: "Devices", icon: "RadioTower" },
    ],
  },
};

/* ─── Fixed: deployment targets ─────────────────────────────────────────────────
   Rendered as "Deployment Options". Values trace to GET /deployment/targets and
   the matching generator; ESP32 is excluded platform-wide by decision.

   `bestFor` states intended usage and renders only on the Integration pages
   (hardwareMode "full"). It is where a target's real limits belong: the C++ and
   Arduino generators accept classification models only — they reject FOMO and
   box-detection models outright — so those two must never be described in a way
   that implies detection.
   ---------------------------------------------------------------------------- */

export const HARDWARE_TARGETS: Record<
  HardwareTargetId,
  { name: string; pkg: string; runs: string; bestFor: string; icon: IconName; accent: string }
> = {
  tflite: {
    name: "TensorFlow Lite",
    pkg: "TensorFlow Lite model with an inference library",
    runs: "Embed it in any runtime that can load TensorFlow Lite",
    bestFor: "Adding classification or detection to an application you already ship.",
    icon: "Binary", accent: "#f59e0b",
  },
  cpp: {
    name: "C++ Library",
    pkg: "Portable C++17 inference library",
    runs: "Compile it into your own firmware or application",
    bestFor: "Classification models on hardware you build the firmware for.",
    icon: "Code2", accent: "#3b82f6",
  },
  arduino: {
    name: "Arduino Library",
    pkg: "Arduino library, ready to import",
    runs: "Import it through the Arduino Library Manager",
    bestFor: "Classification models on a microcontroller board.",
    icon: "CircuitBoard", accent: "#08979d",
  },
  raspberry_pi: {
    name: "Raspberry Pi",
    pkg: "Python inference package",
    runs: "Installs with the device client and runs as a service",
    bestFor: "Camera or sensor inference on a Pi you manage from the dashboard.",
    icon: "Cherry", accent: "#e11f52",
  },
  unoq: {
    name: "UNO Q",
    pkg: "Linux runtime package",
    runs: "The on-device runner takes it from there, no glue code to write",
    bestFor: "Shipping a self-contained model package to a Linux device.",
    icon: "PlugZap", accent: "#6366f1",
  },
  pxe: {
    name: "PXE Runtime",
    pkg: "Sealed binary package",
    runs: "Pairs with a Linux target such as UNO Q or Raspberry Pi",
    bestFor: "Shipping a model when the model file itself should not travel with it.",
    icon: "Package", accent: "#8b5cf6",
  },
};

/** Every marketed target, in the order Integration pages present them. */
export const ALL_TARGETS: HardwareTargetId[] = [
  "raspberry_pi", "unoq", "tflite", "cpp", "arduino", "pxe",
];

/**
 * The three targets an Application or Industry page shows. A reader who needs
 * the complete set, or a target's constraints, goes to Integrations.
 */
export const FOCUSED_TARGETS: HardwareTargetId[] = ["raspberry_pi", "unoq", "tflite"];

/* ─── Fixed: why Petal Edge + closing CTA (identical on all 9 pages) ────────── */

export const WHY_PETAL_EDGE = {
  eyebrow: "Why Petal Edge",
  title: "One platform, ",
  titleAccent: "start to finish",
  lede: "Data, labels, training, evaluation, post-processing, deployment and the fleet live in one place, with nothing to stitch together.",
  points: [
    {
      icon: "Layers", accent: "#8b5cf6",
      title: "The whole lifecycle",
      desc: "Collect, label, train, evaluate, post-process, deploy and monitor without leaving the platform.",
    },
    {
      icon: "Radar", accent: "#6366f1",
      title: "Models sized for edge hardware",
      desc: "NanoVision, Vision Pro and MobileNetV2 are built to run on devices, not on datacentre GPUs.",
    },
    {
      icon: "Package", accent: "#3b82f6",
      title: "Six deployment targets",
      desc: "One trained model, packaged for the hardware you already run.",
    },
    {
      icon: "RadioTower", accent: "#22c55e",
      title: "A fleet you can reach",
      desc: "Register devices, read their inference logs, and push new models over the air.",
    },
  ] as { icon: IconName; accent: string; title: string; desc: string }[],
  included: [
    "Dataset import from COCO, YOLO, Pascal VOC, Open Images and Edge Impulse JSON",
    "AI-assisted labeling with human review before anything is applied",
    "Confusion matrix, per-class metrics and model testing across versions",
    "Post-processing with NMS and IoU tracking",
    "Over-the-air model updates to registered devices",
    "Project export, so your data and models stay portable",
  ],
};

/** Hub page copy + metadata. Deliberately distinct from WHY_PETAL_EDGE.lede,
    which also renders on this page — the two must not say the same thing. */
export const HUB = {
  eyebrow: "Solutions",
  title: "Edge AI, built for ",
  titleAccent: "what you actually ship",
  lede: "Nine pages, three questions: what you can build, where teams put it to work, and how a trained model reaches the hardware.",
  seo: {
    title: "Solutions — Petal Edge",
    description:
      "Edge AI solutions by application, industry and deployment target. See what you can build, where it applies, and how a trained model reaches your hardware.",
  },
};

export const CTA_ACTIONS: { label: string; href: string; variant: "primary" | "ghost" }[] = [
  { label: "Start Free", href: "/login/", variant: "primary" },
  { label: "Request Demo", href: "/book-demo/", variant: "ghost" },
  { label: "Contact Sales", href: "/contact/", variant: "ghost" },
];

/* ─── The nine solutions ────────────────────────────────────────────────────── */

export const SOLUTIONS: Solution[] = [
  /* ---- Applications ------------------------------------------------------ */
  {
    slug: "computer-vision",
    group: "applications",
    name: "Computer Vision",
    tagline: "Classification and detection models, from raw dataset to running device.",
    hero: {
      title: "Computer vision that ships to ",
      titleAccent: "real hardware",
      subtitle:
        "Train image classification and object detection models on your own data, then build a package for the device that has to run them.",
      /* Three chips, one per question a reader arrives with: what can it learn,
         what does it train, where does the result run. */
      chips: [
        "Classification and detection",
        "NanoVision · Vision Pro · MobileNetV2",
        "Ships to Raspberry Pi, UNO Q, TensorFlow Lite",
      ],
    },
    problem: {
      statement:
        "Vision projects stall between the notebook and the device. Labels live in one tool, training in another, and nothing at the end produces a package the hardware can actually run.",
      audience: ["ML engineers", "Embedded and firmware teams", "Automation engineers"],
    },
    workflow: ["vision"],
    features: ["F2", "F3", "F4", "F8", "F9", "F16"],
    useCases: [
      { title: "Counting objects on a moving line", desc: "Detect items as they pass a fixed camera and track them across frames.", model: "NanoVision", target: "raspberry_pi" },
      { title: "Presence and absence checks", desc: "Confirm a part is where it should be before the next step runs.", model: "MobileNetV2", target: "unoq" },
      { title: "Part classification", desc: "Separate variants that look alike but differ in detail.", model: "Transfer learning", target: "tflite" },
      { title: "Tracking objects across frames", desc: "IoU tracking keeps one identity per object, so one thing is not counted twice.", model: "Vision Pro", target: "raspberry_pi" },
      { title: "Many small objects per frame", desc: "Centroid detection where boxes cost more than the device can afford.", model: "NanoVision v2", target: "tflite" },
    ],
    hardwareMode: "focused",
    hardware: FOCUSED_TARGETS,
    related: [
      { label: "Quality Inspection", href: "/solutions/quality-inspection/" },
      { label: "Manufacturing", href: "/solutions/manufacturing/" },
      { label: "Raspberry Pi", href: "/solutions/raspberry-pi/" },
    ],
    ctaHeadline: "Start with your own images",
    seo: {
      title: "Computer Vision on Edge Devices — Petal Edge",
      description:
        "Train image classification and object detection models on your own data with NanoVision, Vision Pro and MobileNetV2, then build a package for the device that runs them.",
    },
  },
  {
    slug: "quality-inspection",
    group: "applications",
    name: "Quality Inspection",
    tagline: "Pass/fail and defect-class checks, built as image classification or detection.",
    hero: {
      title: "Inspection models trained on ",
      titleAccent: "your parts",
      subtitle:
        "Turn labelled examples of good and bad output into a classification or detection model, then score it against a held-out test set before it reaches the line.",
      chips: ["Confusion matrix per run", "Model testing across versions", "Dataset health checks"],
      secondary: { label: "See the workflow", href: "#workflow" },
    },
    problem: {
      statement:
        "Hand-written inspection rules break the moment the product changes. A trained model adapts as you add examples, but only if you can measure whether the new one is actually better than the model it replaces.",
      audience: ["Quality engineers", "Process and production engineers", "ML engineers"],
    },
    workflow: ["vision"],
    features: ["F1", "F7", "F9", "F11", "F12", "F19"],
    useCases: [
      { title: "Pass/fail surface check", desc: "Classify a captured surface as acceptable or not.", model: "MobileNetV2", target: "raspberry_pi" },
      { title: "Defect-type classification", desc: "Sort defects into the classes you already track.", model: "Transfer learning", target: "unoq" },
      { title: "Component presence check", desc: "Detect whether every expected component is in place.", model: "NanoVision", target: "tflite" },
      { title: "Label and print verification", desc: "Check that the right label is applied and readable.", model: "MobileNetV2", target: "tflite" },
      { title: "Locating the defect, not just flagging it", desc: "Return a box around what failed so an operator can see why.", model: "Vision Pro", target: "raspberry_pi" },
    ],
    hardwareMode: "focused",
    hardware: FOCUSED_TARGETS,
    related: [
      { label: "Computer Vision", href: "/solutions/computer-vision/" },
      { label: "Manufacturing", href: "/solutions/manufacturing/" },
      { label: "UNO Q", href: "/solutions/uno-q/" },
    ],
    ctaHeadline: "Train an inspection model on your data",
    seo: {
      title: "Quality Inspection Models — Petal Edge",
      description:
        "Turn labelled examples of good and bad output into a classification or detection model, and score it against a held-out test set before it reaches the line.",
    },
  },
  {
    slug: "safety-monitoring",
    group: "applications",
    name: "Safety Monitoring",
    tagline: "Build your own safety-gear detector from your own footage.",
    hero: {
      title: "Build a safety detector on ",
      titleAccent: "your own footage",
      subtitle:
        "Petal Edge does not ship a pre-trained safety model. You label footage from your own site, train a detector on it, tune the thresholds against a real clip, then deploy.",
      chips: ["AI-assisted labeling", "Threshold and class filters", "IoU tracking"],
      secondary: { label: "See the workflow", href: "#workflow" },
    },
    problem: {
      statement:
        "An off-the-shelf safety model was trained on someone else's site, with their cameras, angles and lighting. Yours are different, which is why the detector has to be trained on your footage.",
      audience: ["EHS and site safety teams", "Systems integrators", "ML engineers"],
    },
    workflow: ["vision"],
    features: ["F3", "F4", "F8", "F13", "F14", "F18"],
    useCases: [
      { title: "Helmet check at an entry gate", desc: "Detect whether head protection is present as people pass a fixed camera.", model: "Vision Pro", target: "raspberry_pi" },
      { title: "High-visibility vest detection", desc: "Trained on your own yard footage, in your own lighting.", model: "Vision Pro", target: "unoq" },
      { title: "Glove detection at a workstation", desc: "Small-object detection close to the work surface.", model: "NanoVision v2", target: "tflite" },
      { title: "Counting entries into a monitored area", desc: "IoU tracking keeps one identity per person, so one crossing counts once.", model: "Vision Pro", target: "raspberry_pi" },
      { title: "Tuning alert sensitivity before rollout", desc: "Move the confidence threshold, render a real clip, and see the effect first.", model: "Vision Pro", target: "unoq" },
    ],
    hardwareMode: "focused",
    hardware: FOCUSED_TARGETS,
    related: [
      { label: "Computer Vision", href: "/solutions/computer-vision/" },
      { label: "Manufacturing", href: "/solutions/manufacturing/" },
      { label: "Raspberry Pi", href: "/solutions/raspberry-pi/" },
    ],
    ctaHeadline: "Label your footage and train a detector",
    seo: {
      title: "Safety Monitoring — Build Your Own Detector — Petal Edge",
      description:
        "Petal Edge ships no pre-trained safety model. Label footage from your own site, train a detector on it, tune the thresholds against a real clip, then deploy.",
    },
  },

  /* ---- Industries -------------------------------------------------------- */
  {
    slug: "manufacturing",
    group: "industries",
    name: "Manufacturing",
    tagline: "Line-side vision and machine-signal models on one platform.",
    hero: {
      title: "Vision and machine signals, ",
      titleAccent: "one platform",
      subtitle:
        "Cameras on the line and sensors on the machine feed the same pipeline: the same dataset tools, the same training jobs, the same deployment targets.",
      chips: ["Image and time-series models", "Spectral, MFCC and spectrogram blocks", "Over-the-air updates"],
      secondary: { label: "See both pipelines", href: "#workflow" },
    },
    problem: {
      statement:
        "A plant rarely has one kind of data. Vision problems and vibration or sound problems end up in separate tools, with separate deployment stories. You bring the data; the pipeline is the same for both.",
      audience: ["Automation and controls engineers", "Quality engineers", "Plant IT and OT teams"],
    },
    workflow: ["vision", "sensor"],
    features: ["F5", "F9", "F10", "F16", "F17", "F18"],
    useCases: [
      { title: "Line-side pass/fail", desc: "Classify captured frames as the product moves through the station.", model: "MobileNetV2", target: "raspberry_pi" },
      { title: "Part counting", desc: "Count items with centroid detection instead of boxes.", model: "NanoVision", target: "unoq" },
      { title: "Vibration-signature classification", desc: "Spectral analysis features into a 1D convolutional network.", model: "1D convolutional network", target: "raspberry_pi" },
      { title: "Machine-sound classification", desc: "MFCC features into a classifier trained on your own recordings.", model: "2D convolutional network", target: "tflite" },
      { title: "Machine state from a sensor sequence", desc: "Classify a window of readings rather than a single sample.", model: "LSTM network", target: "tflite" },
      { title: "Catching a defect the model has not seen", desc: "Add fresh samples and retrain from the existing impulse.", model: "Transfer learning", target: "unoq" },
    ],
    hardwareMode: "focused",
    hardware: FOCUSED_TARGETS,
    related: [
      { label: "Quality Inspection", href: "/solutions/quality-inspection/" },
      { label: "Safety Monitoring", href: "/solutions/safety-monitoring/" },
      { label: "UNO Q", href: "/solutions/uno-q/" },
    ],
    ctaHeadline: "Bring your line data",
    seo: {
      title: "Manufacturing — Vision and Machine Signals — Petal Edge",
      description:
        "Cameras on the line and sensors on the machine feed one pipeline: the same dataset tools, the same training jobs, the same deployment targets.",
    },
  },
  {
    slug: "agriculture",
    group: "industries",
    name: "Agriculture",
    tagline: "Vision models that keep working where there is no connectivity.",
    hero: {
      title: "Models that run ",
      titleAccent: "in the field",
      subtitle:
        "Train on the images your equipment already captures, then deploy to hardware that runs the model locally instead of depending on a connection.",
      chips: ["Runs without a connection", "NanoVision centroid detection", "Raspberry Pi and UNO Q"],
      secondary: { label: "See the workflow", href: "#workflow" },
    },
    problem: {
      statement:
        "Field sites rarely have reliable connectivity, and sending video back for processing is not an option. You bring the imagery; the model runs on the device that captured it.",
      audience: ["Agritech engineering teams", "Equipment manufacturers", "Research and agronomy teams"],
    },
    workflow: ["vision"],
    features: ["F1", "F3", "F8", "F15", "F16", "F17"],
    useCases: [
      { title: "Crop versus weed classification", desc: "Classify what the camera sees as it passes over a row.", model: "MobileNetV2", target: "raspberry_pi" },
      { title: "Counting with centroids", desc: "Many small objects per frame, without paying for boxes.", model: "NanoVision", target: "unoq" },
      { title: "Pest presence detection", desc: "Detect whether a target is present in a captured frame.", model: "Vision Pro", target: "raspberry_pi" },
      { title: "Livestock counting", desc: "Count animals in a frame and hold the identities across frames.", model: "NanoVision", target: "tflite" },
      { title: "Grading harvested produce", desc: "Sort what passes the camera into the grades you already use.", model: "Transfer learning", target: "raspberry_pi" },
    ],
    hardwareMode: "focused",
    hardware: FOCUSED_TARGETS,
    related: [
      { label: "Computer Vision", href: "/solutions/computer-vision/" },
      { label: "Retail", href: "/solutions/retail/" },
      { label: "Raspberry Pi", href: "/solutions/raspberry-pi/" },
    ],
    ctaHeadline: "Train on your field imagery",
    seo: {
      title: "Agriculture — Vision Models That Run Offline — Petal Edge",
      description:
        "Train on the images your equipment already captures, then deploy to hardware that runs the model locally instead of depending on a connection.",
    },
  },
  {
    slug: "retail",
    group: "industries",
    name: "Retail",
    tagline: "On-device analytics without sending customer video anywhere.",
    hero: {
      title: "Store analytics that stay ",
      titleAccent: "in the store",
      subtitle:
        "Detection models run on hardware in the store. Frames are processed locally, and what leaves the device is a result, not the footage.",
      chips: ["Inference on local hardware", "IoU tracking", "Video preview before rollout"],
      secondary: { label: "See the workflow", href: "#workflow" },
    },
    problem: {
      statement:
        "Store analytics usually means streaming camera footage somewhere else, which is expensive and hard to justify to anyone. You bring the cameras; the model runs on a device beside them.",
      audience: ["Retail technology teams", "Systems integrators", "Store operations"],
    },
    workflow: ["vision"],
    features: ["F3", "F8", "F13", "F14", "F15", "F16"],
    useCases: [
      { title: "Shelf stock-out detection", desc: "Detect empty facings from a fixed shelf camera.", model: "Vision Pro", target: "raspberry_pi" },
      { title: "Queue-length counting", desc: "Count people in a defined view with stable identities.", model: "NanoVision", target: "unoq" },
      { title: "Product-facing verification", desc: "Check that products are present and oriented as expected.", model: "MobileNetV2", target: "tflite" },
      { title: "Footfall counting", desc: "IoU tracking keeps a person who lingers from being counted twice.", model: "Vision Pro", target: "raspberry_pi" },
      { title: "Planogram spot-checks", desc: "Classify a section against the layout it is meant to match.", model: "Transfer learning", target: "tflite" },
    ],
    hardwareMode: "focused",
    hardware: FOCUSED_TARGETS,
    related: [
      { label: "Computer Vision", href: "/solutions/computer-vision/" },
      { label: "Agriculture", href: "/solutions/agriculture/" },
      { label: "UNO Q", href: "/solutions/uno-q/" },
    ],
    ctaHeadline: "Keep the footage where it is",
    seo: {
      title: "Retail — On-Device Store Analytics — Petal Edge",
      description:
        "Detection models run on hardware in the store. Frames are processed locally, and what leaves the device is a result, not the footage.",
    },
  },

  /* ---- Integrations ------------------------------------------------------ */
  {
    slug: "raspberry-pi",
    group: "integrations",
    name: "Raspberry Pi",
    tagline: "From a trained model to a running Pi service in one build.",
    hero: {
      title: "Deploy to a Pi ",
      titleAccent: "in one build",
      subtitle:
        "Build a Python inference package from any trained model, install it with the device client and its service unit, then manage it from the dashboard.",
      chips: ["Python package build", "systemd service", "Over-the-air updates"],
      secondary: { label: "See the deployment path", href: "#workflow" },
    },
    problem: {
      statement:
        "Getting a model onto a Pi is rarely the hard part. Knowing which build is on which device, and replacing it without a site visit, is.",
      audience: ["Embedded engineers", "Systems integrators", "Field operations teams"],
    },
    workflow: ["integration"],
    features: ["F8", "F15", "F16", "F17", "F18", "F20"],
    useCases: [
      { title: "Camera inference service", desc: "Run detection against the Pi camera as a background service.", model: "Vision Pro", target: "raspberry_pi" },
      { title: "Headless install", desc: "Install through the client script and let the service unit handle restarts.", model: "MobileNetV2", target: "raspberry_pi" },
      { title: "Swapping the model without a site visit", desc: "Request the update from the dashboard instead of connecting to the device.", model: "Any trained model", target: "raspberry_pi" },
      { title: "Counting on a Pi with no accelerator", desc: "Centroid detection sized for the CPU the board actually has.", model: "NanoVision", target: "raspberry_pi" },
      { title: "Shipping the model as a sealed binary", desc: "Use the PXE package when the model file should not travel with it.", model: "Any trained model", target: "pxe" },
    ],
    hardwareMode: "full",
    hardware: ALL_TARGETS,
    related: [
      { label: "Computer Vision", href: "/solutions/computer-vision/" },
      { label: "UNO Q", href: "/solutions/uno-q/" },
      { label: "TensorFlow Lite", href: "/solutions/tensorflow-lite/" },
    ],
    ctaHeadline: "Put a model on your Pi",
    seo: {
      title: "Deploy to Raspberry Pi — Petal Edge",
      description:
        "Build a Python inference package from any trained model, install it with the device client and its service unit, then manage it from the dashboard.",
    },
  },
  {
    slug: "uno-q",
    group: "integrations",
    name: "UNO Q",
    tagline: "A Linux .pe runtime package driven by a manifest.",
    hero: {
      title: "A runtime package for ",
      titleAccent: "UNO Q",
      subtitle:
        "Build a runtime package the on-device runner knows how to execute, or ship a sealed binary instead of the model file itself.",
      chips: ["Linux runtime package", "Sealed binary option", "Profile-matched deployments"],
      secondary: { label: "See the deployment path", href: "#workflow" },
    },
    problem: {
      statement:
        "Shipping a model to a Linux edge board usually means hand-writing the glue: preprocessing, tensor layout, output decoding. The package should carry that itself, so the device does not need a bespoke integration.",
      audience: ["Embedded Linux engineers", "Product teams shipping devices", "Systems integrators"],
    },
    workflow: ["integration"],
    features: ["F13", "F15", "F16", "F17", "F18", "F20"],
    useCases: [
      { title: "Detection with no glue code", desc: "Preprocessing and output decoding travel inside the package.", model: "Vision Pro", target: "unoq" },
      { title: "Shipping a sealed binary", desc: "Use the PXE package when the model file should not travel with it.", model: "Any trained model", target: "pxe" },
      { title: "Profile-matched deployment", desc: "The device resolves which build it is compatible with before it updates.", model: "Any trained model", target: "unoq" },
      { title: "Streaming live inference", desc: "Watch results in the dashboard while the model runs on the board.", model: "NanoVision", target: "unoq" },
      { title: "Staged rollout", desc: "Request an update per device and check its status before the next one.", model: "Any trained model", target: "unoq" },
    ],
    hardwareMode: "full",
    hardware: ALL_TARGETS,
    related: [
      { label: "Raspberry Pi", href: "/solutions/raspberry-pi/" },
      { label: "TensorFlow Lite", href: "/solutions/tensorflow-lite/" },
      { label: "Manufacturing", href: "/solutions/manufacturing/" },
    ],
    ctaHeadline: "Build a package for UNO Q",
    seo: {
      title: "Deploy to Arduino UNO Q — Petal Edge",
      description:
        "Build a Linux runtime package the on-device runner executes without glue code, or ship a sealed binary instead of the model file itself.",
    },
  },
  {
    slug: "tensorflow-lite",
    group: "integrations",
    name: "TensorFlow Lite",
    tagline: "Take the .tflite and integrate it anywhere.",
    hero: {
      title: "Your model as a ",
      titleAccent: ".tflite file",
      subtitle:
        "Build a TensorFlow Lite model with an inference library, or take the portable C++17 library or the Arduino package instead. No runtime to adopt.",
      chips: [".tflite + inference library", "Portable C++17 library", "Arduino library"],
      secondary: { label: "See the deployment path", href: "#workflow" },
    },
    problem: {
      statement:
        "Most teams already have an application. They do not need another runtime to adopt. They need a model file, and enough code around it to call.",
      audience: ["Application developers", "Firmware engineers", "ML engineers"],
    },
    workflow: ["integration"],
    features: ["F8", "F9", "F11", "F12", "F16", "F20"],
    useCases: [
      { title: "Dropping a model into an existing app", desc: "Take the .tflite build and call it from code you already ship.", model: "MobileNetV2", target: "tflite" },
      { title: "Embedding on custom hardware", desc: "Compile the portable C++17 library into your own firmware.", model: "Transfer learning", target: "cpp" },
      { title: "Calling a model from a sketch", desc: "Import the generated library and classify a sensor buffer on the board.", model: "1D convolutional network", target: "arduino" },
      { title: "Detection in your own pipeline", desc: "Take the detection model and decode its output where you want it.", model: "Vision Pro", target: "tflite" },
      { title: "Counting without boxes", desc: "Centroid detection where a box per object costs more than it is worth.", model: "NanoVision", target: "tflite" },
    ],
    hardwareMode: "full",
    hardware: ALL_TARGETS,
    related: [
      { label: "Raspberry Pi", href: "/solutions/raspberry-pi/" },
      { label: "UNO Q", href: "/solutions/uno-q/" },
      { label: "Computer Vision", href: "/solutions/computer-vision/" },
    ],
    ctaHeadline: "Build your first TensorFlow Lite model",
    seo: {
      title: "TensorFlow Lite Export — Petal Edge",
      description:
        "Build a TensorFlow Lite model with an inference library, or take the portable C++17 library or the Arduino package instead. No runtime to adopt.",
    },
  },
];

/* ─── Lookups ───────────────────────────────────────────────────────────────── */

export const SOLUTION_SLUGS = SOLUTIONS.map((s) => s.slug);

export function getSolution(slug: string): Solution | undefined {
  return SOLUTIONS.find((s) => s.slug === slug);
}

export function solutionsByGroup(group: SolutionGroup): Solution[] {
  return SOLUTIONS.filter((s) => s.group === group);
}

/** Canonical path for a solution — trailingSlash is on in next.config.js. */
export function solutionHref(slug: string): string {
  return `/solutions/${slug}/`;
}
