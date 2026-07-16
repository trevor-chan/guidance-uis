(function () {
  "use strict";

  const STORAGE_KEY = "poseGuidanceExperimentSession.v1";
  const DATA_CATEGORIES = Object.freeze(["real", "practice", "trash"]);
  const DEFAULT_DATA_CATEGORY = "practice";

  function newSessionId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  const MODALITIES = Object.freeze({
    M1: Object.freeze({
      id: "M1",
      display: "1D",
      frame: "none",
      label: "1D visual display",
      shortLabel: "1D",
      page: "index.html",
      needsCalibration: false,
    }),
    M2: Object.freeze({
      id: "M2",
      display: "2D",
      frame: "user",
      label: "2D - User reference frame",
      shortLabel: "2D / User",
      page: "index-2d.html",
      needsCalibration: true,
    }),
    M3: Object.freeze({
      id: "M3",
      display: "2D",
      frame: "patient",
      label: "2D - Patient reference frame",
      shortLabel: "2D / Patient",
      page: "index-2d.html",
      needsCalibration: false,
    }),
    M4: Object.freeze({
      id: "M4",
      display: "2D",
      frame: "transducer",
      label: "2D - Transducer reference frame",
      shortLabel: "2D / Transducer",
      page: "index-2d.html",
      needsCalibration: false,
    }),
    M5: Object.freeze({
      id: "M5",
      display: "3D",
      frame: "user",
      label: "3D - User reference frame",
      shortLabel: "3D / User",
      page: "index-3d.html",
      needsCalibration: true,
    }),
    M6: Object.freeze({
      id: "M6",
      display: "3D",
      frame: "patient",
      label: "3D - Patient reference frame",
      shortLabel: "3D / Patient",
      page: "index-3d.html",
      needsCalibration: false,
    }),
    M7: Object.freeze({
      id: "M7",
      display: "3D",
      frame: "transducer",
      label: "3D - Transducer reference frame",
      shortLabel: "3D / Transducer",
      page: "index-3d.html",
      needsCalibration: false,
    }),
    M8: Object.freeze({
      id: "M8",
      display: "3D",
      frame: "hybrid",
      label: "3D - Hybrid reference frame",
      shortLabel: "3D / Hybrid",
      page: "index-3d.html",
      // Hybrid's camera stays fixed like transducer's — calibration is only used
      // to draw the calibrated-view-axis line in the scene, not to move the camera.
      // See study/sequence.py needs_calibration and index-3d.html's
      // updateHybridCalibrationAxis.
      needsCalibration: true,
    }),
  });

  // Rows P1-P7 from the protocol's condition matrix. Each position is the
  // modality assigned to target set S1-S7, respectively.
  const OPTION_ORDERS = Object.freeze([
    Object.freeze(["M1", "M2", "M3", "M4", "M5", "M6", "M7"]),
    Object.freeze(["M7", "M1", "M2", "M3", "M4", "M5", "M6"]),
    Object.freeze(["M6", "M7", "M1", "M2", "M3", "M4", "M5"]),
    Object.freeze(["M5", "M6", "M7", "M1", "M2", "M3", "M4"]),
    Object.freeze(["M4", "M5", "M6", "M7", "M1", "M2", "M3"]),
    Object.freeze(["M3", "M4", "M5", "M6", "M7", "M1", "M2"]),
    Object.freeze(["M2", "M3", "M4", "M5", "M6", "M7", "M1"]),
  ]);

  const MODALITY_TRIALS_PER_CONDITION = 3;

  // TEMPORARY: learning_curve mode list/order. Change here if the study
  // design shifts. IDs reference MODALITIES above (M3 = 2D/Patient,
  // M5 = 3D/User); order is fixed for every participant (no rotation).
  const LEARNING_CURVE_MODES = ["M3", "M5"];
  const LEARNING_CURVE_TRIALS_PER_MODE = 20;

  // TEMPORARY: NOISE mode list/order and magnitude ramp. Change here if the
  // study design shifts. M3 = 2D/Patient, M5 = 3D/User.
  const NOISE_MODES = ["M3", "M5"];
  const NOISE_MAGNITUDES = [0, 1, 2, 4, 8];   // mm position AND degrees orientation
  const NOISE_TRIALS_PER_MAGNITUDE = 3;

  // TEMPORARY: LATENCY mode list/order and delay ramp. Change here if the
  // study design shifts. M3 = 2D/Patient, M5 = 3D/User.
  const LATENCY_MODES = ["M3", "M5"];
  // Injected delay only; this is what the server buffers on top of the
  // ~75ms measured system baseline. See LATENCY_BASELINE_MS for the
  // perceived (injected + baseline) value that gets persisted alongside it.
  const LATENCY_MS = [0, 75, 225, 525, 1125];
  const LATENCY_BASELINE_MS = 75;
  const LATENCY_TRIALS_PER_MAGNITUDE = 3;

  // TEMPORARY: PRECISION mode list/order and match-threshold ramp (easiest ->
  // hardest per mode). Change here if the study design shifts. M3 = 2D/Patient,
  // M5 = 3D/User.
  const PRECISION_MODES = ["M3", "M5"];
  const PRECISION_THRESHOLDS = [
    { linearMm: 10, angularDeg: 10 },
    { linearMm: 5,  angularDeg: 5  },
    { linearMm: 3,  angularDeg: 3  },
    { linearMm: 1,  angularDeg: 1  },
  ];
  const PRECISION_TRIALS_PER_THRESHOLD = 3;

  // TEMPORARY: HYBRID mode list and trial count. Change here if the study
  // design shifts. M8 = 3D/Hybrid (position-locked camera, live rotation —
  // see index-3d.html:1202-1239).
  const HYBRID_MODES = ["M8"];
  const HYBRID_TRIALS = 20; // placeholder — easy to change

  function participantOption(participantId) {
    const match = String(participantId || "").trim().match(/^p?([1-7])$/i);
    return match ? Number(match[1]) : null;
  }

  function buildModalityConditions(optionNumber) {
    const order = OPTION_ORDERS[optionNumber - 1];
    if (!order) throw new Error("Condition option must be between 1 and 7.");
    return order.map((modalityId, index) => ({
      index,
      targetSet: `S${index + 1}`,
      modalityId,
      nTrials: MODALITY_TRIALS_PER_CONDITION,
      status: "pending",
    }));
  }

  function buildLearningCurveConditions() {
    return LEARNING_CURVE_MODES.map((modalityId, index) => ({
      index,
      targetSet: null,
      modalityId,
      nTrials: LEARNING_CURVE_TRIALS_PER_MODE,
      status: "pending",
    }));
  }

  // Fisher-Yates shuffle. Uses Math.random() (no seed) so the trial order is
  // fresh per participant, per the scrambled-block design (see
  // buildNoiseConditions/buildLatencyConditions/buildPrecisionConditions).
  function shuffled(items) {
    const arr = items.slice();
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  // Noise/latency/precision each pool every magnitude/threshold x its
  // per-level repeat count into ONE scrambled block per mode (instead of one
  // block per magnitude): the magnitude becomes a per-trial property, carried
  // in condition.trialPlan (one entry per trial, in run order). Practice runs
  // once at the start of that block, hence includePractice is always true.
  function buildNoiseConditions() {
    return NOISE_MODES.map((modalityId, index) => {
      const plan = [];
      NOISE_MAGNITUDES.forEach(magnitude => {
        for (let i = 0; i < NOISE_TRIALS_PER_MAGNITUDE; i++) plan.push({ noise: magnitude });
      });
      return {
        index,
        targetSet: null,
        modalityId,
        nTrials: plan.length,
        trialPlan: shuffled(plan),
        includePractice: true,
        status: "pending",
      };
    });
  }

  function buildLatencyConditions() {
    return LATENCY_MODES.map((modalityId, index) => {
      const plan = [];
      LATENCY_MS.forEach(latencyMs => {
        for (let i = 0; i < LATENCY_TRIALS_PER_MAGNITUDE; i++) {
          plan.push({ latencyMs, perceivedMs: latencyMs + LATENCY_BASELINE_MS });
        }
      });
      return {
        index,
        targetSet: null,
        modalityId,
        nTrials: plan.length,
        trialPlan: shuffled(plan),
        includePractice: true,
        status: "pending",
      };
    });
  }

  function buildPrecisionConditions() {
    return PRECISION_MODES.map((modalityId, index) => {
      const plan = [];
      PRECISION_THRESHOLDS.forEach(threshold => {
        for (let i = 0; i < PRECISION_TRIALS_PER_THRESHOLD; i++) {
          plan.push({
            precisionLinearMm: threshold.linearMm,
            precisionAngularDeg: threshold.angularDeg,
          });
        }
      });
      return {
        index,
        targetSet: null,
        modalityId,
        nTrials: plan.length,
        trialPlan: shuffled(plan),
        includePractice: true,
        status: "pending",
      };
    });
  }

  function buildHybridConditions() {
    return HYBRID_MODES.map((modalityId, index) => ({
      index,
      targetSet: null,
      modalityId,
      nTrials: HYBRID_TRIALS,
      includePractice: true,
      status: "pending",
    }));
  }

  // Registry of experiment types. Each entry supplies a buildConditions()
  // function and the flags a session of that type runs with.
  const EXPERIMENTS = Object.freeze({
    modality: {
      buildConditions: buildModalityConditions,
      flags: Object.freeze({ practice: true, preference: true, pausable: false }),
      usesMatrixOption: true,
    },
    learning_curve: {
      buildConditions: buildLearningCurveConditions,
      flags: Object.freeze({ practice: true, preference: false, pausable: true }),
      usesMatrixOption: false,
    },
    noise: {
      buildConditions: buildNoiseConditions,
      flags: Object.freeze({ practice: true, preference: false, pausable: true }),
      usesMatrixOption: false,
    },
    latency: {
      buildConditions: buildLatencyConditions,
      flags: Object.freeze({ practice: true, preference: false, pausable: true }),
      usesMatrixOption: false,
    },
    precision: {
      buildConditions: buildPrecisionConditions,
      flags: Object.freeze({ practice: true, preference: false, pausable: true }),
      usesMatrixOption: false,
    },
    hybrid: {
      buildConditions: buildHybridConditions,
      flags: Object.freeze({ practice: true, preference: false, pausable: true }),
      usesMatrixOption: false,
    },
  });

  function buildConditions(optionNumber) {
    // Retained for backward compatibility: always the modality-matrix build.
    return buildModalityConditions(optionNumber);
  }

  function createSession(participantId, optionNumber, metadata = {}) {
    const normalizedId = String(participantId || "").trim().toUpperCase();
    if (!normalizedId) throw new Error("Enter a participant ID.");

    const dataCategory = DATA_CATEGORIES.includes(metadata.dataCategory)
      ? metadata.dataCategory
      : DEFAULT_DATA_CATEGORY;
    const experimentCondition = metadata.experimentCondition || "modality";
    const experiment = EXPERIMENTS[experimentCondition];
    if (!experiment) throw new Error(`Unknown experiment condition: ${experimentCondition}`);
    if (!experiment.buildConditions) {
      throw new Error(`Experiment "${experimentCondition}" is not implemented yet.`);
    }

    let option = null;
    let conditions;
    if (experiment.usesMatrixOption) {
      option = Number(optionNumber);
      const linkedOption = participantOption(normalizedId);
      if (!OPTION_ORDERS[option - 1]) throw new Error("Select condition option 1-7.");
      if (linkedOption && linkedOption !== option) {
        throw new Error(`Participant ${normalizedId} is linked to Option ${linkedOption}.`);
      }
      conditions = experiment.buildConditions(option);
    } else {
      conditions = experiment.buildConditions();
    }

    const session = {
      version: 2,
      sessionId: newSessionId(),
      participantId: normalizedId,
      participantName: String(metadata.participantName || "").trim(),
      examinerName: String(metadata.examinerName || "").trim(),
      experimentCondition,
      dataCategory,
      option,
      flags: experiment.flags,
      conditions,
      currentIndex: 0,
      boxSet: false,
      startedAt: new Date().toISOString(),
      completedAt: null,
      persistentState: null,
    };
    saveSession(session);
    return session;
  }

  function saveSession(session) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    return session;
  }

  function getSession() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (!parsed || !Array.isArray(parsed.conditions) || !parsed.participantId) return null;
      if (!parsed.sessionId) {
        parsed.sessionId = newSessionId();
        parsed.version = 2;
        parsed.experimentCondition ||= "modality";
        parsed.participantName ||= "";
        parsed.examinerName ||= "";
        saveSession(parsed);
      }
      return parsed;
    } catch (_error) {
      return null;
    }
  }

  function clearSession() {
    localStorage.removeItem(STORAGE_KEY);
  }

  function markBoxSet() {
    const session = getSession();
    if (!session) return null;
    session.boxSet = true;
    return saveSession(session);
  }

  function currentCondition(session = getSession()) {
    if (!session || session.currentIndex >= session.conditions.length) return null;
    return session.conditions[session.currentIndex];
  }

  function conditionDetails(condition) {
    if (!condition) return null;
    return {
      ...condition,
      modality: MODALITIES[condition.modalityId],
    };
  }

  function conditionUrl(session, condition = currentCondition(session)) {
    const details = conditionDetails(condition);
    if (!session || !details) return "launcher.html";
    const params = new URLSearchParams({
      study: "1",
      frame: details.modality.frame,
      participant: session.participantId,
      option: session.option != null ? String(session.option) : "",
      condition: String(details.index + 1),
      target_set: details.targetSet || "",
      modality_id: details.modalityId,
      session_id: session.sessionId,
      n_trials: String(details.nTrials || MODALITY_TRIALS_PER_CONDITION),
      include_practice: details.includePractice === false ? "0" : "1",
    });
    return `${details.modality.page}?${params.toString()}`;
  }

  function startBlockMetadata() {
    const context = contextForLocation();
    if (!context) return {};
    const flags = context.session?.flags || { practice: true, preference: true, pausable: false };
    return {
      participant_id: context.participantId,
      session_id: context.sessionId,
      condition_option: context.option,
      condition_index: context.conditionNumber,
      target_set: context.targetSet || null,
      modality_id: context.modalityId,
      n_trials: context.nTrials,
      include_preference: flags.preference,
      // Per-trial noise/latency/precision ramp for the scrambled noise,
      // latency, and precision experiments (see buildNoiseConditions et al.)
      // — one entry per trial, in the shuffled run order. null for
      // experiments that don't vary per trial (modality, learning_curve).
      trial_plan: context.trialPlan,
      include_practice: context.includePractice,
    };
  }

  function contextForLocation() {
    const params = new URLSearchParams(window.location.search);
    const session = getSession();
    const conditionNumber = Number(params.get("condition"));
    const conditionIndex = Number.isInteger(conditionNumber) && conditionNumber >= 1
      ? conditionNumber - 1
      : session?.currentIndex;
    const condition = session?.conditions?.[conditionIndex];
    const modalityId = params.get("modality_id") || condition?.modalityId;
    const modality = MODALITIES[modalityId];
    if (!modalityId || !modality) return null;
    const nTrialsParam = Number(params.get("n_trials"));
    return {
      session,
      participantId: params.get("participant") || session?.participantId || "",
      sessionId: params.get("session_id") || session?.sessionId || "",
      option: Number(params.get("option")) || session?.option || null,
      conditionIndex,
      conditionNumber: conditionIndex + 1,
      totalConditions: session?.conditions?.length || null,
      targetSet: params.get("target_set") || condition?.targetSet || "",
      nTrials: (Number.isInteger(nTrialsParam) && nTrialsParam > 0)
        ? nTrialsParam
        : (condition?.nTrials || MODALITY_TRIALS_PER_CONDITION),
      trialPlan: condition?.trialPlan || null,
      includePractice: params.has("include_practice")
        ? params.get("include_practice") !== "0"
        : (condition?.includePractice ?? true),
      modalityId,
      modality,
    };
  }

  function renderContext(element) {
    if (!element) return;
    const context = contextForLocation();
    if (!context) {
      element.style.display = "none";
      return;
    }
    element.textContent = [
      context.participantId,
      context.option != null ? `Option ${context.option}` : null,
      `Condition ${context.conditionNumber} of ${context.totalConditions || "?"}`,
      context.targetSet ? `${context.targetSet} / ${context.modalityId}` : context.modalityId,
      context.modality.label,
    ].filter(Boolean).join("  |  ");
  }

  function completeConditionFromLocation() {
    const session = getSession();
    if (!session) return null;

    const params = new URLSearchParams(window.location.search);
    const requested = Number(params.get("condition")) - 1;
    const index = Number.isInteger(requested) && requested >= 0
      ? requested
      : session.currentIndex;
    const condition = session.conditions[index];
    if (condition) {
      condition.status = "complete";
      condition.completedAt = condition.completedAt || new Date().toISOString();
    }

    const nextIndex = session.conditions.findIndex(item => item.status !== "complete");
    session.currentIndex = nextIndex === -1 ? session.conditions.length : nextIndex;
    if (nextIndex === -1) session.completedAt = session.completedAt || new Date().toISOString();
    return saveSession(session);
  }

  function persistencePayload(session = getSession()) {
    if (!session) return null;
    return {
      session_id: session.sessionId,
      participant_id: session.participantId,
      participant_name: session.participantName,
      examiner_name: session.examinerName,
      experiment_condition: session.experimentCondition,
      condition_option: session.option,
      data_category: session.dataCategory || DEFAULT_DATA_CATEGORY,
      started_at: session.startedAt,
      metadata: {
        client_version: session.version,
      },
      conditions: session.conditions.map(condition => {
        const modality = MODALITIES[condition.modalityId];
        return {
          condition_index: condition.index + 1,
          target_set: condition.targetSet,
          modality_id: condition.modalityId,
          modality: modality.display.toLowerCase(),
          frame: modality.frame,
          noise: condition.noiseMagnitude ?? null,
          latency_ms: condition.latencyMs ?? null,
          perceived_ms: condition.perceivedMs ?? null,
          learning_curve: null,
          precision_linear_mm: condition.precisionLinearMm ?? null,
          precision_angular_deg: condition.precisionAngularDeg ?? null,
        };
      }),
    };
  }

  function applyPersistentState(persistentState) {
    const session = getSession();
    if (!session || !persistentState?.session) return session;
    const byIndex = new Map(
      (persistentState.conditions || []).map(condition => [
        Number(condition.condition_index) - 1,
        condition,
      ])
    );
    session.conditions.forEach(condition => {
      const saved = byIndex.get(condition.index);
      if (!saved) return;
      condition.status = saved.status || condition.status;
      condition.completedTrials = Number(saved.completed_trials || 0);
      condition.completedAt = saved.completed_at || condition.completedAt || null;
    });
    session.persistentState = {
      databasePath: persistentState.database_path,
      storageBackend: persistentState.storage_backend,
      storageLayout: persistentState.storage_layout,
      exportFormats: persistentState.export_formats || [],
    };
    session.boxSet = Boolean(persistentState.has_box_pose);
    if (persistentState.session.completed_at) {
      session.completedAt = persistentState.session.completed_at;
    }
    const firstIncomplete = session.conditions.findIndex(
      condition => condition.status !== "complete"
    );
    if (
      session.currentIndex >= session.conditions.length ||
      session.conditions[session.currentIndex]?.status === "complete"
    ) {
      session.currentIndex = firstIncomplete === -1
        ? session.conditions.length
        : firstIncomplete;
    }
    return saveSession(session);
  }

  function restorePersistentSession(persistentState) {
    if (!persistentState?.session) return null;
    const savedSession = persistentState.session;
    const experimentCondition = savedSession.experiment_condition || "modality";
    const experiment = EXPERIMENTS[experimentCondition];
    // Re-derive nTrials/flags from the registry (not persisted in the DB
    // conditions table) by rebuilding the same template this experiment
    // type would produce, then overlaying it on the saved condition rows.
    let template = [];
    try {
      if (experiment?.buildConditions) {
        template = experiment.usesMatrixOption
          ? experiment.buildConditions(Number(savedSession.condition_option))
          : experiment.buildConditions();
      }
    } catch (_error) {
      template = [];
    }
    // Noise/latency/precision no longer carry a single per-condition
    // magnitude (see buildNoiseConditions et al.) — the per-trial ramp lives
    // only in trialPlan, which isn't persisted to the DB. A resumed session
    // gets a freshly reshuffled trialPlan from the template; this only
    // matters for a condition resumed on a different browser/machine than
    // the one that started it (localStorage normally carries the original
    // trialPlan through page reloads on the same machine untouched).
    const conditions = (persistentState.conditions || []).map((condition, index) => ({
      index,
      targetSet: condition.target_set || null,
      modalityId: condition.modality_id,
      nTrials: template[index]?.nTrials || MODALITY_TRIALS_PER_CONDITION,
      trialPlan: template[index]?.trialPlan ?? null,
      includePractice: template[index]?.includePractice ?? true,
      status: condition.status,
      completedTrials: Number(condition.completed_trials || 0),
      completedAt: condition.completed_at,
    }));
    const firstIncomplete = conditions.findIndex(condition => condition.status !== "complete");
    const session = {
      version: 2,
      sessionId: savedSession.session_id,
      participantId: savedSession.participant_id,
      participantName: savedSession.participant_name || "",
      examinerName: savedSession.examiner_name || "",
      experimentCondition,
      option: savedSession.condition_option,
      flags: experiment?.flags || { practice: true, preference: true, pausable: false },
      conditions,
      currentIndex: firstIncomplete === -1 ? conditions.length : firstIncomplete,
      boxSet: Boolean(persistentState.has_box_pose),
      startedAt: savedSession.started_at,
      completedAt: savedSession.completed_at,
      persistentState: {
        databasePath: persistentState.database_path,
        storageBackend: persistentState.storage_backend,
        storageLayout: persistentState.storage_layout,
        exportFormats: persistentState.export_formats || [],
      },
    };
    return saveSession(session);
  }

  function selectCondition(index) {
    const session = getSession();
    if (!session || !session.conditions[index]) return session;
    session.currentIndex = index;
    return saveSession(session);
  }

  function completeAndReturnToDashboard() {
    completeConditionFromLocation();
    window.location.replace("launcher.html?resume=1");
  }

  // ── Inter-trial countdown ────────────────────────────────────────────────
  // Full-screen, opaque, always-on-top overlay so the participant can see
  // neither the previous nor the upcoming target while it's up. Shared by all
  // three modality pages so the occlusion behavior is identical everywhere.
  const COUNTDOWN_OVERLAY_ID = "trial-countdown-overlay";
  let _countdownShownForTrial = null;
  let _countdownBusy = false;

  function countdownOverlay() {
    let overlay = document.getElementById(COUNTDOWN_OVERLAY_ID);
    if (overlay) return overlay;
    overlay = document.createElement("div");
    overlay.id = COUNTDOWN_OVERLAY_ID;
    Object.assign(overlay.style, {
      position: "fixed",
      inset: "0",
      zIndex: "999999",
      display: "none",
      alignItems: "center",
      justifyContent: "center",
      background: "#000",
      color: "#fff",
      fontFamily: "'Courier New', monospace",
      fontWeight: "bold",
      fontSize: "min(45vh, 45vw)",
    });
    document.body.appendChild(overlay);
    return overlay;
  }

  function runTrialCountdown(onComplete) {
    const overlay = countdownOverlay();
    overlay.style.display = "flex";
    let count = 3;
    overlay.textContent = String(count);
    const tick = () => {
      count -= 1;
      if (count <= 0) {
        overlay.style.display = "none";
        onComplete?.();
        return;
      }
      overlay.textContent = String(count);
      setTimeout(tick, 1000);
    };
    setTimeout(tick, 1000);
  }

  // Call every render tick while activity_type is "trial". No-ops unless
  // trialIndex is one we haven't already counted down for, and guards against
  // re-triggering from the ~30Hz stream of messages while the countdown runs.
  function maybeRunTrialCountdown(trialIndex, onComplete) {
    if (trialIndex === null || trialIndex === undefined) return;
    if (trialIndex === _countdownShownForTrial || _countdownBusy) return;
    _countdownBusy = true;
    runTrialCountdown(() => {
      _countdownShownForTrial = trialIndex;
      _countdownBusy = false;
      onComplete?.();
    });
  }

  window.ExperimentSession = Object.freeze({
    MODALITIES,
    OPTION_ORDERS,
    EXPERIMENTS,
    LEARNING_CURVE_MODES,
    HYBRID_MODES,
    DATA_CATEGORIES,
    DEFAULT_DATA_CATEGORY,
    participantOption,
    buildConditions,
    createSession,
    saveSession,
    getSession,
    clearSession,
    markBoxSet,
    currentCondition,
    conditionDetails,
    conditionUrl,
    startBlockMetadata,
    contextForLocation,
    renderContext,
    completeConditionFromLocation,
    completeAndReturnToDashboard,
    persistencePayload,
    applyPersistentState,
    restorePersistentSession,
    selectCondition,
    runTrialCountdown,
    maybeRunTrialCountdown,
  });
})();
