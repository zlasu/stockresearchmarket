<template>
  <section v-if="detail && variant" class="page">
    <header class="topbar">
      <div>
        <RouterLink class="back-link" :to="`/experiments/${detail.id}`">Back to experiment</RouterLink>
        <p class="eyebrow">{{ variant.kind }}</p>
        <h1>{{ variant.name }}</h1>
      </div>
    </header>

    <div class="kpi-grid">
      <KpiCard label="CAGR" :value="formatMetric('cagr', variant.metrics.cagr)" />
      <KpiCard label="Sharpe" :value="formatMetric('sharpe', variant.metrics.sharpe)" />
      <KpiCard label="Max DD" :value="formatMetric('drawdown_magnitude', variant.metrics.drawdown_magnitude)" />
      <KpiCard label="Turnover" :value="formatMetric('avg_annual_turnover', variant.metrics.avg_annual_turnover)" />
    </div>

    <div class="split-grid">
      <ChartCard title="Equity" subtitle="Variant equity curve." :option="equityOption" />
      <ChartCard title="Drawdown" subtitle="Computed or stored drawdown series." :option="drawdownOption" />
    </div>

    <section class="panel">
      <header class="panel-header">
        <div>
          <h2>Parameters</h2>
          <p>Original run configuration fields captured by the indexer.</p>
        </div>
      </header>
      <DataTable :rows="paramRows" :keys="['key', 'value']" />
    </section>

    <section v-if="table" class="panel">
      <header class="panel-header">
        <div>
          <h2>Trades</h2>
          <p>First rows from the local trade artifact.</p>
        </div>
      </header>
      <DataTable :rows="table.rows" :keys="table.columns.map((column) => column.key).slice(0, 10)" />
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";

import { api } from "../api";
import ChartCard from "../components/ChartCard.vue";
import DataTable from "../components/DataTable.vue";
import KpiCard from "../components/KpiCard.vue";
import { singleLineOption } from "../charts";
import { formatMetric } from "../lib/format";
import { useResearchStore } from "../stores/research";
import type { SeriesPoint, TablePayload, VariantSummary } from "../types";

const route = useRoute();
const store = useResearchStore();
const id = computed(() => String(route.params.id));
const variantId = computed(() => String(route.params.variantId));
const variant = ref<VariantSummary | null>(null);
const equity = ref<SeriesPoint[]>([]);
const drawdown = ref<SeriesPoint[]>([]);
const table = ref<TablePayload | null>(null);
const detail = computed(() => store.details[id.value]);
const equityOption = computed(() => singleLineOption(equity.value, "Equity"));
const drawdownOption = computed(() => singleLineOption(drawdown.value, "Drawdown"));
const paramRows = computed(() =>
  Object.entries(variant.value?.params ?? {}).map(([key, value]) => ({
    key,
    value: typeof value === "object" ? JSON.stringify(value) : value,
  })),
);

onMounted(async () => {
  await store.loadExperiment(id.value);
  variant.value = await api.variant(id.value, variantId.value);
  equity.value = (await api.series(id.value, variantId.value, "equity")).rows;
  drawdown.value = (await api.series(id.value, variantId.value, "drawdown")).rows;
  if (variant.value.available_tables.includes("trades")) {
    table.value = await api.table(id.value, "trades", variantId.value);
  }
});
</script>

