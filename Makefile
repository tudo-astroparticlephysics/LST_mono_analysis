OUTDIR = build
OBSDIR = observations
SIMDIR = simulations


AICT_CONFIG = config/aict.yaml
CUTS_CONFIG = config/quality_cuts.yaml
CUTS_CONFIG_DATA = $(CUTS_CONFIG)
TEL_NAME = LST_LSTCam

OBS_VERSION=v0.6.1_v05
SIM_VERSION=20201023_v0.6.3_prod5_local_wo_n_islands


GAMMA_FILE = gamma_20deg_180deg_off0.4deg_$(SIM_VERSION)
GAMMA_DIFFUSE_FILE = gamma-diffuse_20deg_180deg_$(SIM_VERSION)
PROTON_FILE = proton_20deg_180deg_$(SIM_VERSION)
ELECTRON_FILE = electron_20deg_180deg_$(SIM_VERSION)


CRAB_RUNS=2766 2767 2768 2769 2770 2771
MRK421_RUNS=2113 2114 2115 2116 2117 2130 2131 2132 2133
MRK501_RUNS=2606 2607 2608 2610 2612 2613

CRAB_DL2=$(addsuffix .h5, $(addprefix $(OUTDIR)/dl2_$(OBS_VERSION)_LST-1.Run0, $(CRAB_RUNS)))
MRK421_DL2=$(addsuffix .h5, $(addprefix $(OUTDIR)/dl2_$(OBS_VERSION)_LST-1.Run0, $(MRK421_RUNS)))
MRK501_DL2=$(addsuffix .h5, $(addprefix $(OUTDIR)/dl2_$(OBS_VERSION)_LST-1.Run0, $(MRK501_RUNS)))


all: $(OUTDIR)/cv_separation.h5 \
	$(OUTDIR)/cv_disp.h5 \
	$(OUTDIR)/cv_regressor.h5 \
	$(OUTDIR)/regressor_plots.pdf \
	$(OUTDIR)/disp_plots.pdf \
	$(OUTDIR)/separator_plots.pdf \
	$(OUTDIR)/dl2_$(GAMMA_FILE)_testing.h5 \
	$(OUTDIR)/dl2_$(GAMMA_DIFFUSE_FILE)_testing.h5 \
	$(OUTDIR)/dl2_$(PROTON_FILE)_testing.h5 \
	$(OUTDIR)/dl2_$(ELECTRON_FILE)_testing.h5 \
	$(CRAB_DL2) \
	$(MRK421_DL2) \
	$(MRK501_DL2) \
	$(OUTDIR)/crab_theta2.pdf \
	$(OUTDIR)/mrk421_theta2.pdf \
	$(OUTDIR)/mrk501_theta2.pdf \
	$(OUTDIR)/pyirf.fits.gz

#file convert
$(OUTDIR)/%_aict.h5: $(OBSDIR)/%.h5 file_convert.py | $(OUTDIR)
	python file_convert.py \
		$< \
		$@ \
		$(TEL_NAME)

$(OUTDIR)/%_aict.h5: $(SIMDIR)/%.h5 file_convert.py | $(OUTDIR)
	python file_convert.py \
		$< \
		$@ \
		$(TEL_NAME)


#precuts
$(OUTDIR)/%_precuts.h5: $(OUTDIR)/%_aict.h5 $(CUTS_CONFIG) | $(OUTDIR)
	aict_apply_cuts \
		$(CUTS_CONFIG) \
		$< \
		$@

#train models
$(OUTDIR)/separator.pkl $(OUTDIR)/cv_separation.h5: $(CUTS_CONFIG) $(AICT_CONFIG) $(OUTDIR)/dl1_$(PROTON_FILE)_training_precuts.h5
$(OUTDIR)/separator.pkl $(OUTDIR)/cv_separation.h5: $(OUTDIR)/dl1_$(GAMMA_DIFFUSE_FILE)_training_precuts.h5
	aict_train_separation_model \
		$(AICT_CONFIG) \
		$(OUTDIR)/dl1_$(GAMMA_DIFFUSE_FILE)_training_precuts.h5 \
		$(OUTDIR)/dl1_$(PROTON_FILE)_training_precuts.h5 \
		$(OUTDIR)/cv_separation.h5 \
		$(OUTDIR)/separator.pkl

$(OUTDIR)/disp.pkl $(OUTDIR)/sign.pkl $(OUTDIR)/cv_disp.h5: $(CUTS_CONFIG) $(AICT_CONFIG) $(OUTDIR)/dl1_$(GAMMA_DIFFUSE_FILE)_training_precuts.h5
	aict_train_disp_regressor \
		$(AICT_CONFIG) \
		$(OUTDIR)/dl1_$(GAMMA_DIFFUSE_FILE)_training_precuts.h5 \
		$(OUTDIR)/cv_disp.h5 \
		$(OUTDIR)/disp.pkl \
		$(OUTDIR)/sign.pkl

$(OUTDIR)/regressor.pkl $(OUTDIR)/cv_regressor.h5: $(CUTS_CONFIG) $(AICT_CONFIG) $(OUTDIR)/dl1_$(GAMMA_FILE)_training_precuts.h5
	aict_train_energy_regressor \
		$(AICT_CONFIG) \
		$(OUTDIR)/dl1_$(GAMMA_FILE)_training_precuts.h5 \
		$(OUTDIR)/cv_regressor.h5 \
		$(OUTDIR)/regressor.pkl

#apply models
$(OUTDIR)/dl2_%.h5: $(OUTDIR)/dl1_%_aict.h5 $(OUTDIR)/separator.pkl $(OUTDIR)/disp.pkl $(OUTDIR)/regressor.pkl $(AICT_CONFIG) $(CUTS_CONFIG_DATA) add_az_alt.py
	aict_apply_cuts \
		$(CUTS_CONFIG_DATA) \
		$< $@ \
		--chunksize=100000
	aict_apply_separation_model \
		$(AICT_CONFIG) \
		$@ \
		$(OUTDIR)/separator.pkl \
		--chunksize=100000
	aict_apply_disp_regressor \
		$(AICT_CONFIG) \
		$@ \
		$(OUTDIR)/disp.pkl \
		$(OUTDIR)/sign.pkl \
		--chunksize=100000
	aict_apply_energy_regressor \
		$(AICT_CONFIG) \
		$@ \
		$(OUTDIR)/regressor.pkl \
		--chunksize=100000
	python add_az_alt.py \
		$@

#performance plots
$(OUTDIR)/regressor_plots.pdf: $(AICT_CONFIG) $(OUTDIR)/cv_regressor.h5 | $(OUTDIR)
	aict_plot_regressor_performance \
		$(AICT_CONFIG) \
		$(OUTDIR)/cv_regressor.h5 \
		$(OUTDIR)/regressor.pkl \
		-o $@

$(OUTDIR)/separator_plots.pdf: $(AICT_CONFIG) $(OUTDIR)/cv_separation.h5 | $(OUTDIR)
	aict_plot_separator_performance \
		$(AICT_CONFIG) \
		$(OUTDIR)/cv_separation.h5 \
		$(OUTDIR)/separator.pkl \
		-o $@

$(OUTDIR)/disp_plots.pdf: $(AICT_CONFIG) $(OUTDIR)/cv_disp.h5 $(OUTDIR)/dl1_$(GAMMA_DIFFUSE_FILE)_training_precuts.h5 | $(OUTDIR)
	aict_plot_disp_performance \
		$(AICT_CONFIG) \
		$(OUTDIR)/cv_disp.h5 \
		$(OUTDIR)/dl1_$(GAMMA_DIFFUSE_FILE)_training_precuts.h5 \
		$(OUTDIR)/disp.pkl \
		$(OUTDIR)/sign.pkl \
		-o $@

#observations
$(OUTDIR)/crab_theta2.pdf: theta2_wobble.py plotting.py $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02766.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02767.h5 $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02768.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02769.h5 $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02770.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02771.h5 | $(OUTDIR)
	python theta2_wobble.py \
		$(OUTDIR)/crab_theta2.pdf \
		'Crab' \
		0.03 \
		0.85 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02766.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02767.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02768.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02769.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02770.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02771.h5

$(OUTDIR)/mrk421_theta2.pdf: theta2_wobble.py plotting.py $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02113.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02114.h5 $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02115.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02116.h5 $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02117.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02130.h5 $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02131.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02132.h5 $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02133.h5 | $(OUTDIR)
	python theta2_wobble.py \
		$(OUTDIR)/mrk421_theta2.pdf \
		'Mrk 421' \
		0.03 \
		0.85 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02113.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02114.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02115.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02116.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02117.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02130.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02131.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02132.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02133.h5

$(OUTDIR)/mrk501_theta2.pdf: theta2_wobble.py plotting.py $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02606.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02607.h5 $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02608.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02610.h5 $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02612.h5 \
  $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02613.h5 | $(OUTDIR)
	python theta2_wobble.py \
		$(OUTDIR)/mrk501_theta2.pdf \
		'Mrk 501' \
		0.03 \
		0.85 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02606.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02607.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02608.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02610.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02612.h5 \
		-d $(OUTDIR)/dl2_v0.6.1_v05_LST-1.Run02613.h5

#pyirf sensitivity 
$(OUTDIR)/pyirf.fits.gz: pyirf_sensitivity.py $(OUTDIR)/dl2_$(GAMMA_FILE)_testing.h5 $(OUTDIR)/dl2_$(PROTON_FILE)_testing.h5 $(OUTDIR)/dl2_$(ELECTRON_FILE)_testing.h5 | $(OUTDIR)
	python pyirf_sensitivity.py \
		$(OUTDIR)/dl2_$(GAMMA_FILE)_testing.h5 \
		$(OUTDIR)/dl2_$(PROTON_FILE)_testing.h5 \
		$(OUTDIR)/dl2_$(ELECTRON_FILE)_testing.h5 \
		$(OUTDIR)/pyirf.fits.gz


$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf $(OUTDIR)


.PHONY: all clean
