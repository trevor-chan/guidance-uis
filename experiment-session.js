(function () {
  "use strict";

  const STORAGE_KEY = "poseGuidanceExperimentSession.v1";

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

  function participantOption(participantId) {
    const match = String(participantId || "").trim().match(/^p?([1-7])$/i);
    return match ? Number(match[1]) : null;
  }

  function buildConditions(optionNumber) {
    const order = OPTION_ORDERS[optionNumber - 1];
    if (!order) throw new Error("Condition option must be between 1 and 7.");
    return order.map((modalityId, index) => ({
      index,
      targetSet: `S${index + 1}`,
      modalityId,
      status: "pending",
    }));
  }

  function createSession(participantId, optionNumber, metadata = {}) {
    const normalizedId = String(participantId || "").trim().toUpperCase();
    const option = Number(optionNumber);
    const linkedOption = participantOption(normalizedId);
    if (!normalizedId) throw new Error("Enter a participant ID.");
    if (!OPTION_ORDERS[option - 1]) throw new Error("Select condition option 1-7.");
    if (linkedOption && linkedOption !== option) {
      throw new Error(`Participant ${normalizedId} is linked to Option ${linkedOption}.`);
    }

    const session = {
      version: 2,
      sessionId: newSessionId(),
      participantId: normalizedId,
      participantName: String(metadata.participantName || "").trim(),
      examinerName: String(metadata.examinerName || "").trim(),
      experimentCondition: metadata.experimentCondition || "modality",
      option,
      conditions: buildConditions(option),
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
      option: String(session.option),
      condition: String(details.index + 1),
      target_set: details.targetSet,
      modality_id: details.modalityId,
      session_id: session.sessionId,
    });
    return `${details.modality.page}?${params.toString()}`;
  }

  function startBlockMetadata() {
    const context = contextForLocation();
    if (!context) return {};
    return {
      participant_id: context.participantId,
      session_id: context.sessionId,
      condition_option: context.option,
      condition_index: context.conditionNumber,
      target_set: context.targetSet,
      modality_id: context.modalityId,
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
    return {
      participantId: params.get("participant") || session?.participantId || "",
      sessionId: params.get("session_id") || session?.sessionId || "",
      option: Number(params.get("option")) || session?.option || null,
      conditionIndex,
      conditionNumber: conditionIndex + 1,
      targetSet: params.get("target_set") || condition?.targetSet || "",
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
      `Option ${context.option}`,
      `Condition ${context.conditionNumber} of 7`,
      `${context.targetSet} / ${context.modalityId}`,
      context.modality.label,
    ].join("  |  ");
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
          noise: null,
          latency_ms: null,
          learning_curve: null,
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
    const conditions = (persistentState.conditions || []).map((condition, index) => ({
      index,
      targetSet: condition.target_set,
      modalityId: condition.modality_id,
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
      experimentCondition: savedSession.experiment_condition || "modality",
      option: savedSession.condition_option,
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

  window.ExperimentSession = Object.freeze({
    MODALITIES,
    OPTION_ORDERS,
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
  });
})();
