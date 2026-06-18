import { defineStore } from "pinia";

import { api } from "../api";
import type { ComparePayload, ExperimentDetail, ExperimentSummary, ScatterPayload } from "../types";

interface ResearchState {
  experiments: ExperimentSummary[];
  total: number;
  loading: boolean;
  error: string | null;
  details: Record<string, ExperimentDetail>;
  scatters: Record<string, ScatterPayload>;
  comparisons: Record<string, ComparePayload>;
}

export const useResearchStore = defineStore("research", {
  state: (): ResearchState => ({
    experiments: [],
    total: 0,
    loading: false,
    error: null,
    details: {},
    scatters: {},
    comparisons: {},
  }),
  actions: {
    async loadExperiments(params: Record<string, string | number | undefined> = {}) {
      this.loading = true;
      this.error = null;
      try {
        const payload = await api.experiments(params);
        this.experiments = payload.items;
        this.total = payload.total;
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
      } finally {
        this.loading = false;
      }
    },
    async refresh() {
      await api.refresh();
      await this.loadExperiments();
    },
    async loadExperiment(id: string) {
      if (this.details[id]) return this.details[id];
      this.loading = true;
      this.error = null;
      try {
        const detail = await api.experiment(id);
        this.details[id] = detail;
        return detail;
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
        throw error;
      } finally {
        this.loading = false;
      }
    },
    async loadScatter(id: string) {
      if (this.scatters[id]) return this.scatters[id];
      const scatter = await api.scatter(id);
      this.scatters[id] = scatter;
      return scatter;
    },
    async loadCompare(id: string, variantIds: string[]) {
      const key = `${id}:${variantIds.join(",")}`;
      if (this.comparisons[key]) return this.comparisons[key];
      const compare = await api.compare(id, variantIds);
      this.comparisons[key] = compare;
      return compare;
    },
  },
});

