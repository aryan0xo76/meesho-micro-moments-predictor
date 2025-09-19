// Meesho Micro-Moment Prediction Engine JavaScript
class MeeshoApp {
  constructor() {
    this.state = {
      dataGenerated: false,
      modelsTrained: false,
      isGeneratingData: false,
      isTrainingModels: false,
      isGeneratingRecommendations: false,
      trainingProgress: 0,
    };

    this.personas = [
      {
        id: "tier2_fashion",
        name: "Tier-2 Fashion Family Shopper",
        description:
          "Women's ethnic wear focus, evening engagement peak, Diwali/wedding season boost",
        icon: "👗",
      },
      {
        id: "student_examprep",
        name: "Campus Student Exam-Prep",
        description:
          "Stationery and budget electronics focus, late-night engagement, exam window boost",
        icon: "📚",
      },
      {
        id: "budget_gadget",
        name: "Budget Gadget Seeker",
        description:
          "Low-cost electronics, weekend timing, festival deal sensitivity",
        icon: "📱",
      },
      {
        id: "home_decor_festive",
        name: "Home Decor Festive Upgrader",
        description:
          "Decor and lighting before Diwali, afternoon engagement pattern",
        icon: "🏠",
      },
      {
        id: "regional_festive",
        name: "Regional Festive Wear (Bengal Focus)",
        description:
          "Sarees and puja items during Durga Puja, morning browsing pattern",
        icon: "🎭",
      },
    ];

    this.init();
  }

  init() {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => {
        this.bindEvents();
        this.populatePersonas();
        this.enforceWorkflowOrder();
        this.checkSystemStatus();
      });
    } else {
      this.bindEvents();
      this.populatePersonas();
      this.enforceWorkflowOrder();
      this.checkSystemStatus();
    }
  }

  bindEvents() {
    const generateBtn = document.getElementById("generateDataBtn");
    const trainBtn = document.getElementById("trainModelsBtn");
    const recommendBtn = document.getElementById("generateRecommendationsBtn");
    const copyBtn = document.getElementById("copyMessageBtn");

    if (generateBtn)
      generateBtn.addEventListener("click", () => this.generateData());
    if (trainBtn) trainBtn.addEventListener("click", () => this.trainModels());
    if (recommendBtn)
      recommendBtn.addEventListener("click", () =>
        this.generateRecommendations()
      );
    if (copyBtn)
      copyBtn.addEventListener("click", () => this.copyWhatsAppMessage());
  }

  // FIXED: WORKFLOW ENFORCEMENT FUNCTION
  enforceWorkflowOrder() {
    console.log("Enforcing workflow order...", this.state);

    const trainingCard = document.getElementById("trainingCard");
    const recommendationsCard = document.getElementById("recommendationsCard");
    const trainBtn = document.getElementById("trainModelsBtn");
    const recommendBtn = document.getElementById("generateRecommendationsBtn");
    const personaSelect = document.getElementById("personaSelect");

    // Step 2: Training (only if data generated)
    if (!this.state.dataGenerated) {
      if (trainingCard) trainingCard.classList.add("step-card--disabled");
      if (trainBtn) {
        trainBtn.disabled = true;
        trainBtn.style.cursor = "not-allowed";
        trainBtn.style.opacity = "0.5";
      }
    } else {
      // FIX: Remove disabled class when data is generated
      if (trainingCard) trainingCard.classList.remove("step-card--disabled");
      if (trainBtn) {
        trainBtn.disabled = false;
        trainBtn.style.cursor = "pointer";
        trainBtn.style.opacity = "1";
      }
    }

    // Step 3: Recommendations (only if models trained)
    if (!this.state.modelsTrained) {
      if (recommendationsCard)
        recommendationsCard.classList.add("step-card--disabled");
      if (recommendBtn) {
        recommendBtn.disabled = true;
        recommendBtn.style.cursor = "not-allowed";
        recommendBtn.style.opacity = "0.5";
      }
      if (personaSelect) {
        personaSelect.disabled = true;
        personaSelect.style.cursor = "not-allowed";
        personaSelect.style.opacity = "0.5";
      }
    } else {
      // FIX: Remove disabled class when models are trained
      if (recommendationsCard)
        recommendationsCard.classList.remove("step-card--disabled");
      if (recommendBtn) {
        recommendBtn.disabled = false;
        recommendBtn.style.cursor = "pointer";
        recommendBtn.style.opacity = "1";
      }
      if (personaSelect) {
        personaSelect.disabled = false;
        personaSelect.style.cursor = "pointer";
        personaSelect.style.opacity = "1";
      }
    }
  }

  populatePersonas() {
    const select = document.getElementById("personaSelect");
    if (!select) return;

    select.innerHTML = '<option value="">Choose a persona...</option>';

    this.personas.forEach((persona) => {
      const option = document.createElement("option");
      option.value = persona.id;
      option.textContent = `${persona.icon} ${persona.name}`;
      option.title = persona.description;
      select.appendChild(option);
    });
  }

  async checkSystemStatus() {
    try {
      const response = await fetch("/api/status");
      if (response.ok) {
        const status = await response.json();
        this.state.dataGenerated = status.data_generated;
        this.state.modelsTrained = status.models_trained;

        if (this.state.dataGenerated) {
          this.updateStatusBadge("dataStatus", "Data: Generated", "success");
        }
        if (this.state.modelsTrained) {
          this.updateStatusBadge("modelStatus", "Models: Trained", "success");
        }

        this.enforceWorkflowOrder();
        this.showToast("System status loaded", "success");
      }
    } catch (error) {
      this.showToast("Using offline mode", "warning");
      this.enforceWorkflowOrder();
    }
  }

  async generateData() {
    if (this.state.isGeneratingData) return;

    this.state.isGeneratingData = true;
    this.updateGenerateDataButton(true);

    try {
      const nUsers = parseInt(document.getElementById("nUsers").value);
      const nProducts = parseInt(document.getElementById("nProducts").value);
      const days = parseInt(document.getElementById("days").value);

      const response = await fetch("/api/generate-data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          n_users: nUsers,
          n_products: nProducts,
          days: days,
          seed: 42,
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();

      // UPDATE STATE TO ENABLE NEXT STEP
      this.state.dataGenerated = true;

      this.showDataResults({
        files_created: result.files?.length || 5,
        users_count: nUsers,
        products_count: nProducts,
        interactions_count: Math.floor(nUsers * nProducts * 0.1),
      });

      this.updateStatusBadge("dataStatus", "Data: Generated", "success");
      this.showToast("✅ Data generated! You can now train models.", "success");
    } catch (error) {
      this.showToast("❌ Failed to generate data: " + error.message, "error");
    } finally {
      this.state.isGeneratingData = false;
      this.updateGenerateDataButton(false);
      this.enforceWorkflowOrder(); // Re-enforce after state change
    }
  }

  async trainModels() {
    // STRICT VALIDATION
    if (!this.state.dataGenerated) {
      this.showToast(
        "❌ Please generate data first before training models!",
        "error"
      );
      return;
    }

    if (this.state.isTrainingModels) return;

    this.state.isTrainingModels = true;
    this.updateTrainModelsButton(true);
    this.showTrainingProgress();

    try {
      const response = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          models: ["covisitation", "sto", "reranker", "headlines"],
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      // Poll for training status
      while (true) {
        const statusResponse = await fetch("/api/training-status");
        const status = await statusResponse.json();

        this.updateTrainingProgress(status.message, status.progress);

        if (status.status === "completed") {
          this.state.trainingProgress = 100;
          break;
        } else if (status.status === "error") {
          throw new Error(status.message);
        }

        await this.sleep(1000);
      }

      // UPDATE STATE TO ENABLE NEXT STEP
      this.state.modelsTrained = true;

      this.updateStatusBadge("modelStatus", "Models: Trained", "success");
      this.showToast(
        "✅ Models trained! You can now generate recommendations.",
        "success"
      );
    } catch (error) {
      this.showToast("❌ Failed to train models: " + error.message, "error");
    } finally {
      this.state.isTrainingModels = false;
      this.updateTrainModelsButton(false);
      this.enforceWorkflowOrder(); // Re-enforce after state change
    }
  }

  async generateRecommendations() {
    // STRICT VALIDATION FOR ALL PREREQUISITES
    if (!this.state.dataGenerated) {
      this.showToast("❌ Generate data first!", "error");
      return;
    }

    if (!this.state.modelsTrained) {
      this.showToast("❌ Train models first!", "error");
      return;
    }

    const personaSelect = document.getElementById("personaSelect");
    if (!personaSelect || !personaSelect.value) {
      this.showToast("❌ Please select a persona first", "error");
      return;
    }

    if (this.state.isGeneratingRecommendations) return;

    this.state.isGeneratingRecommendations = true;
    this.updateRecommendationsButton(true);

    try {
      const selectedPersona = this.personas.find(
        (p) => p.id === personaSelect.value
      );

      const response = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          persona_id: personaSelect.value,
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const recommendations = await response.json();

      this.displayRecommendations(recommendations, selectedPersona);
      this.showToast("✅ Recommendations generated successfully!", "success");
    } catch (error) {
      this.showToast(
        "❌ Failed to generate recommendations: " + error.message,
        "error"
      );
    } finally {
      this.state.isGeneratingRecommendations = false;
      this.updateRecommendationsButton(false);
    }
  }

  displayRecommendations(recommendations, persona) {
    const headlineEl = document.getElementById("campaignHeadline");
    if (headlineEl) {
      headlineEl.textContent = recommendations.headline;
    }

    const hoursContainer = document.getElementById("optimalHours");
    if (hoursContainer && recommendations.optimal_hours) {
      hoursContainer.innerHTML = recommendations.optimal_hours
        .map((hour) => `<span class="hour-chip">${hour}:00</span>`)
        .join("");
    }

    const productsGrid = document.getElementById("productsGrid");
    if (productsGrid && recommendations.products) {
      productsGrid.innerHTML = recommendations.products
        .slice(0, 6)
        .map(
          (product) => `
                <div class="product-card">
                    <div class="product-image">${persona.icon}</div>
                    <div class="product-info">
                        <div class="product-title">${product.title}</div>
                        <div class="product-price">₹${Math.round(
                          product.price
                        )}</div>
                    </div>
                </div>
            `
        )
        .join("");
    }

    const messageEl = document.getElementById("whatsappMessage");
    if (messageEl) {
      messageEl.innerHTML = recommendations.whatsapp_message.replace(
        /\\n/g,
        "<br>"
      );
    }

    const resultsEl = document.getElementById("recommendationsResults");
    if (resultsEl) {
      resultsEl.classList.remove("hidden");
    }
  }

  copyWhatsAppMessage() {
    const messageElement = document.getElementById("whatsappMessage");
    if (!messageElement) return;

    const message = messageElement.textContent;

    if (navigator.clipboard) {
      navigator.clipboard
        .writeText(message)
        .then(() => {
          this.showToast("Message copied to clipboard!", "success");
          const copyBtn = document.getElementById("copyMessageBtn");
          if (copyBtn) {
            copyBtn.innerHTML = "✅ Copied!";
            setTimeout(() => {
              copyBtn.innerHTML = "📋 Copy Message";
            }, 2000);
          }
        })
        .catch(() => {
          this.fallbackCopyText(message);
        });
    } else {
      this.fallbackCopyText(message);
    }
  }

  fallbackCopyText(text) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.opacity = "0";
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand("copy");
      this.showToast("Message copied to clipboard!", "success");
    } catch (err) {
      this.showToast("Failed to copy message", "error");
    }
    document.body.removeChild(textArea);
  }

  updateGenerateDataButton(loading) {
    const btn = document.getElementById("generateDataBtn");
    const textSpan = btn?.querySelector(".btn-text");
    const loaderSpan = btn?.querySelector(".btn-loader");

    if (loading) {
      if (btn) btn.disabled = true;
      if (textSpan) textSpan.classList.add("hidden");
      if (loaderSpan) loaderSpan.classList.remove("hidden");
    } else {
      if (btn) btn.disabled = false;
      if (textSpan) textSpan.classList.remove("hidden");
      if (loaderSpan) loaderSpan.classList.add("hidden");
    }
  }

  updateTrainModelsButton(loading) {
    const btn = document.getElementById("trainModelsBtn");
    const textSpan = btn?.querySelector(".btn-text");
    const loaderSpan = btn?.querySelector(".btn-loader");

    if (loading) {
      if (btn) btn.disabled = true;
      if (textSpan) textSpan.classList.add("hidden");
      if (loaderSpan) loaderSpan.classList.remove("hidden");
    } else {
      if (btn) btn.disabled = !this.state.dataGenerated;
      if (textSpan) textSpan.classList.remove("hidden");
      if (loaderSpan) loaderSpan.classList.add("hidden");
    }
  }

  updateRecommendationsButton(loading) {
    const btn = document.getElementById("generateRecommendationsBtn");
    const textSpan = btn?.querySelector(".btn-text");
    const loaderSpan = btn?.querySelector(".btn-loader");

    if (loading) {
      if (btn) btn.disabled = true;
      if (textSpan) textSpan.classList.add("hidden");
      if (loaderSpan) loaderSpan.classList.remove("hidden");
    } else {
      if (btn) btn.disabled = !this.state.modelsTrained;
      if (textSpan) textSpan.classList.remove("hidden");
      if (loaderSpan) loaderSpan.classList.add("hidden");
    }
  }

  showDataResults(results) {
    const elements = {
      filesCount: document.getElementById("filesCount"),
      usersCount: document.getElementById("usersCount"),
      productsCount: document.getElementById("productsCount"),
      interactionsCount: document.getElementById("interactionsCount"),
      dataResults: document.getElementById("dataResults"),
    };

    if (elements.filesCount)
      elements.filesCount.textContent = results.files_created;
    if (elements.usersCount)
      elements.usersCount.textContent = results.users_count.toLocaleString();
    if (elements.productsCount)
      elements.productsCount.textContent =
        results.products_count.toLocaleString();
    if (elements.interactionsCount)
      elements.interactionsCount.textContent =
        results.interactions_count.toLocaleString();
    if (elements.dataResults) elements.dataResults.classList.remove("hidden");
  }

  showTrainingProgress() {
    const progressEl = document.getElementById("trainingProgress");
    if (progressEl) {
      progressEl.classList.remove("hidden");
    }
  }

  updateTrainingProgress(text, progress) {
    const textEl = document.getElementById("progressText");
    const fillEl = document.getElementById("progressFill");

    if (textEl) textEl.textContent = text;
    if (fillEl) fillEl.style.width = progress + "%";
  }

  updateStatusBadge(badgeId, text, status) {
    const badge = document.getElementById(badgeId);
    if (!badge) return;

    const icon = badge.querySelector(".status-icon");
    const textSpan = badge.querySelector(".status-text");

    if (textSpan) textSpan.textContent = text;

    badge.classList.remove("status-badge--success", "status-badge--warning");
    if (status === "success") {
      badge.classList.add("status-badge--success");
      if (icon) icon.textContent = "✅";
    } else if (status === "warning") {
      badge.classList.add("status-badge--warning");
      if (icon) icon.textContent = "⚠️";
    }
  }

  showToast(message, type = "info") {
    const toastContainer = document.getElementById("toastContainer");
    if (!toastContainer) return;

    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.textContent = message;

    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("show");
    }, 100);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(100%)";
      setTimeout(() => {
        if (toast.parentNode) {
          toastContainer.removeChild(toast);
        }
      }, 300);
    }, 4000);
  }

  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// Initialize the application
new MeeshoApp();
