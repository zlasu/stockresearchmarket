<template>
  <section v-if="detail && compare" class="page">
    <header class="topbar">
      <div>
        <RouterLink class="back-link" :to="`/experiments/${detail.id}`">Back to experiment</RouterLink>
        <p class="eyebrow">Comparison</p>
        <h1>{{ detail.name }}</h1>
      </div>
    </header>

    <div class="split-grid">
      <ChartCard title="Equity curves" subtitle="Selected variants rebased as stored in experiment artifacts." :option="equityOption" />
      <ChartCard title="Drawdown" subtitle="Underwater curve; lower values mean deeper drawdowns." :option="drawdownOption" />
    </div>

    <section class="panel">
      <header class="panel-header">
        <div>
          <h2>Metrics</h2>
          <p>Shared comparison table.</p>
        </div>
      </header>
      <DataTable :rows="metricRows" :keys="metricKeys" />
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";

import ChartCard from "../components/ChartCard.vue";
import DataTable from "../components/DataTable.vue";
import { compareLineOption } from "../charts";
import { useResearchStore } from "../stores/research";
import type { ComparePayload } from "../types";

const route = useRoute();
const store = useResearchStore();
const id = computed(() => String(route.params.id));
const selected = computed(() => String(route.query.selected ?? "").split(",").filter(Boolean));
const compare = ref<ComparePayload | null>(null);
const detail = computed(() => store.details[id.value]);

const equityOption = computed(() => (compare.value ? compareLineOption(compare.value, "equity", "Equity") : {}));
const drawdownOption = computed(() => (compare.value ? compareLineOption(compare.value, "drawdown", "Drawdown") : {}));
const metricKeys = ["name", "cagr", "sharpe", "max_drawdown", "calmar", "avg_annual_turnover", "trades", "score"];
const metricRows = computed(() =>
  (compare.value?.metrics ?? []).map((row) => ({
    name: row.name,
    cagr: row.metrics.cagr,
    sharpe: row.metrics.sharpe,
    max_drawdown: row.metrics.max_drawdown,
    calmar: row.metrics.calmar,
    avg_annual_turnover: row.metrics.avg_annual_turnover,
    trades: row.metrics.trades,
    score: row.computed_score,
  })),
);

onMounted(async () => {
  const loaded = await store.loadExperiment(id.value);
  const variantIds = selected.value.length ? selected.value : loaded.variants.slice(0, 3).map((variant) => variant.id);
  compare.value = await store.loadCompare(id.value, variantIds);
});
</script>

