"""Bodies for position lookups (#369), from #369's development GETs (2026-10-17).

Provenance: each raw body is retained byte for byte under
``docs/development/reviews/issue369-position-targets/devgets/`` (its README lists the
URLs). ``SPDI_RS429358``, ``SPDI_UNLISTED``, ``SPAN_NC_000019_10_T`` and
``EVA_AT_PIG_DEL`` are those exact bytes. ``GRCH38_MOLECULES``, ``GRCH38_REPORT`` and
``SSCROFA11_MOLECULES`` keep only the fields a position lookup reads; every kept value is
the retained value (``g15``, ``g5``, ``g11``). ``SSCROFA11_REPORT`` is synthetic in the
retained report's shape: no GET of the pig dataset report was made. ``GRCH37_MOLECULES`` is
synthetic too, GRCh37's assembled molecules by their public RefSeq names in the retained report's
shape (no GET of it was made; lengths other than MT's are placeholders). They are observed
source facts, not expectations: tests state their expected outcomes independently.
"""

GRCH38_MOLECULES = (
    b'{"reports":[{"assembly_accession":"GCF_000001405.40","chr_name":"1","ucsc_style_name'
    b'":"chr1","refseq_accession":"NC_000001.11","genbank_accession":"CM000663.2","role":"'
    b'assembled-molecule","sequence_name":"1","length":248956422},{"assembly_accession":"G'
    b'CF_000001405.40","chr_name":"2","ucsc_style_name":"chr2","refseq_accession":"NC_0000'
    b'02.12","genbank_accession":"CM000664.2","role":"assembled-molecule","sequence_name":'
    b'"2","length":242193529},{"assembly_accession":"GCF_000001405.40","chr_name":"3","ucs'
    b'c_style_name":"chr3","refseq_accession":"NC_000003.12","genbank_accession":"CM000665'
    b'.2","role":"assembled-molecule","sequence_name":"3","length":198295559},{"assembly_a'
    b'ccession":"GCF_000001405.40","chr_name":"4","ucsc_style_name":"chr4","refseq_accessi'
    b'on":"NC_000004.12","genbank_accession":"CM000666.2","role":"assembled-molecule","seq'
    b'uence_name":"4","length":190214555},{"assembly_accession":"GCF_000001405.40","chr_na'
    b'me":"5","ucsc_style_name":"chr5","refseq_accession":"NC_000005.10","genbank_accessio'
    b'n":"CM000667.2","role":"assembled-molecule","sequence_name":"5","length":181538259},'
    b'{"assembly_accession":"GCF_000001405.40","chr_name":"6","ucsc_style_name":"chr6","re'
    b'fseq_accession":"NC_000006.12","genbank_accession":"CM000668.2","role":"assembled-mo'
    b'lecule","sequence_name":"6","length":170805979},{"assembly_accession":"GCF_000001405'
    b'.40","chr_name":"7","ucsc_style_name":"chr7","refseq_accession":"NC_000007.14","genb'
    b'ank_accession":"CM000669.2","role":"assembled-molecule","sequence_name":"7","length"'
    b':159345973},{"assembly_accession":"GCF_000001405.40","chr_name":"8","ucsc_style_name'
    b'":"chr8","refseq_accession":"NC_000008.11","genbank_accession":"CM000670.2","role":"'
    b'assembled-molecule","sequence_name":"8","length":145138636},{"assembly_accession":"G'
    b'CF_000001405.40","chr_name":"9","ucsc_style_name":"chr9","refseq_accession":"NC_0000'
    b'09.12","genbank_accession":"CM000671.2","role":"assembled-molecule","sequence_name":'
    b'"9","length":138394717},{"assembly_accession":"GCF_000001405.40","chr_name":"10","uc'
    b'sc_style_name":"chr10","refseq_accession":"NC_000010.11","genbank_accession":"CM0006'
    b'72.2","role":"assembled-molecule","sequence_name":"10","length":133797422},{"assembl'
    b'y_accession":"GCF_000001405.40","chr_name":"11","ucsc_style_name":"chr11","refseq_ac'
    b'cession":"NC_000011.10","genbank_accession":"CM000673.2","role":"assembled-molecule"'
    b',"sequence_name":"11","length":135086622},{"assembly_accession":"GCF_000001405.40","'
    b'chr_name":"12","ucsc_style_name":"chr12","refseq_accession":"NC_000012.12","genbank_'
    b'accession":"CM000674.2","role":"assembled-molecule","sequence_name":"12","length":13'
    b'3275309},{"assembly_accession":"GCF_000001405.40","chr_name":"13","ucsc_style_name":'
    b'"chr13","refseq_accession":"NC_000013.11","genbank_accession":"CM000675.2","role":"a'
    b'ssembled-molecule","sequence_name":"13","length":114364328},{"assembly_accession":"G'
    b'CF_000001405.40","chr_name":"14","ucsc_style_name":"chr14","refseq_accession":"NC_00'
    b'0014.9","genbank_accession":"CM000676.2","role":"assembled-molecule","sequence_name"'
    b':"14","length":107043718},{"assembly_accession":"GCF_000001405.40","chr_name":"15","'
    b'ucsc_style_name":"chr15","refseq_accession":"NC_000015.10","genbank_accession":"CM00'
    b'0677.2","role":"assembled-molecule","sequence_name":"15","length":101991189},{"assem'
    b'bly_accession":"GCF_000001405.40","chr_name":"16","ucsc_style_name":"chr16","refseq_'
    b'accession":"NC_000016.10","genbank_accession":"CM000678.2","role":"assembled-molecul'
    b'e","sequence_name":"16","length":90338345},{"assembly_accession":"GCF_000001405.40",'
    b'"chr_name":"17","ucsc_style_name":"chr17","refseq_accession":"NC_000017.11","genbank'
    b'_accession":"CM000679.2","role":"assembled-molecule","sequence_name":"17","length":8'
    b'3257441},{"assembly_accession":"GCF_000001405.40","chr_name":"18","ucsc_style_name":'
    b'"chr18","refseq_accession":"NC_000018.10","genbank_accession":"CM000680.2","role":"a'
    b'ssembled-molecule","sequence_name":"18","length":80373285},{"assembly_accession":"GC'
    b'F_000001405.40","chr_name":"19","ucsc_style_name":"chr19","refseq_accession":"NC_000'
    b'019.10","genbank_accession":"CM000681.2","role":"assembled-molecule","sequence_name"'
    b':"19","length":58617616},{"assembly_accession":"GCF_000001405.40","chr_name":"20","u'
    b'csc_style_name":"chr20","refseq_accession":"NC_000020.11","genbank_accession":"CM000'
    b'682.2","role":"assembled-molecule","sequence_name":"20","length":64444167},{"assembl'
    b'y_accession":"GCF_000001405.40","chr_name":"21","ucsc_style_name":"chr21","refseq_ac'
    b'cession":"NC_000021.9","genbank_accession":"CM000683.2","role":"assembled-molecule",'
    b'"sequence_name":"21","length":46709983},{"assembly_accession":"GCF_000001405.40","ch'
    b'r_name":"22","ucsc_style_name":"chr22","refseq_accession":"NC_000022.11","genbank_ac'
    b'cession":"CM000684.2","role":"assembled-molecule","sequence_name":"22","length":5081'
    b'8468},{"assembly_accession":"GCF_000001405.40","chr_name":"X","ucsc_style_name":"chr'
    b'X","refseq_accession":"NC_000023.11","genbank_accession":"CM000685.2","role":"assemb'
    b'led-molecule","sequence_name":"X","length":156040895},{"assembly_accession":"GCF_000'
    b'001405.40","chr_name":"Y","ucsc_style_name":"chrY","refseq_accession":"NC_000024.10"'
    b',"genbank_accession":"CM000686.2","role":"assembled-molecule","sequence_name":"Y","l'
    b'ength":57227415},{"assembly_accession":"GCF_000001405.40","chr_name":"MT","ucsc_styl'
    b'e_name":"chrM","refseq_accession":"NC_012920.1","genbank_accession":"J01415.2","role'
    b'":"assembled-molecule","sequence_name":"MT","length":16569}],"total_count":191}'
)
GRCH38_REPORT = (
    b'{"reports":[{"accession":"GCF_000001405.40","paired_accession":"GCA_000001405.29","a'
    b'ssembly_info":{"assembly_name":"GRCh38.p14","synonym":"hg38","assembly_status":"curr'
    b'ent"}}],"total_count":1}'
)
SSCROFA11_MOLECULES = (
    b'{"reports":[{"assembly_accession":"GCF_000003025.6","chr_name":"12","ucsc_style_name'
    b'":"chr12","refseq_accession":"NC_010454.4","genbank_accession":"CM000823.5","role":"'
    b'assembled-molecule","sequence_name":"Chr12","length":61602749}],"total_count":1}'
)
SSCROFA11_REPORT = (
    b'{"reports":[{"accession":"GCF_000003025.6","paired_accession":"GCA_000003025.6","ass'
    b'embly_info":{"assembly_name":"Sscrofa11.1","synonym":"susScr11","assembly_status":"c'
    b'urrent"}}],"total_count":1}'
)
SPDI_RS429358 = b'{"data":{"rsids":[429358]}}'
SPDI_UNLISTED = (
    b'{"error":{"code":404,"message":"SPDI \'NC_000019.10:44908683:TGCGG:A\' does not match '
    b'any RSID"}}'
)
SPAN_NC_000019_10_T = (
    b'{"data":{"seq_id":"NC_000019.10","position":44908683,"deleted_sequence":"T","inserte'
    b'd_sequence":"N"}}'
)
EVA_AT_PIG_DEL = (
    b'[{"accession":5334475172,"version":1,"data":{"assemblyAccession":"GCA_000003025.6","'
    b'taxonomyAccession":9823,"contig":"CM000823.5","start":3556402,"type":"DEL","validate'
    b'd":false,"mapWeight":null,"createdDate":"2023-03-24T15:21:19.475"}}]'
)

# Synthetic: see the module note.
GRCH37_MOLECULES = (
    b'{"reports":[{"assembly_accession":"GCF_000001405.25","chr_name":"1","ucsc_style_name'
    b'":"chr1","refseq_accession":"NC_000001.10","role":"assembled-molecule","sequence_nam'
    b'e":"1","length":300000000},{"assembly_accession":"GCF_000001405.25","chr_name":"2","'
    b'ucsc_style_name":"chr2","refseq_accession":"NC_000002.11","role":"assembled-molecule'
    b'","sequence_name":"2","length":300000000},{"assembly_accession":"GCF_000001405.25","'
    b'chr_name":"3","ucsc_style_name":"chr3","refseq_accession":"NC_000003.11","role":"ass'
    b'embled-molecule","sequence_name":"3","length":300000000},{"assembly_accession":"GCF_'
    b'000001405.25","chr_name":"4","ucsc_style_name":"chr4","refseq_accession":"NC_000004.'
    b'11","role":"assembled-molecule","sequence_name":"4","length":300000000},{"assembly_a'
    b'ccession":"GCF_000001405.25","chr_name":"5","ucsc_style_name":"chr5","refseq_accessi'
    b'on":"NC_000005.9","role":"assembled-molecule","sequence_name":"5","length":300000000'
    b'},{"assembly_accession":"GCF_000001405.25","chr_name":"6","ucsc_style_name":"chr6","'
    b'refseq_accession":"NC_000006.11","role":"assembled-molecule","sequence_name":"6","le'
    b'ngth":300000000},{"assembly_accession":"GCF_000001405.25","chr_name":"7","ucsc_style'
    b'_name":"chr7","refseq_accession":"NC_000007.13","role":"assembled-molecule","sequenc'
    b'e_name":"7","length":300000000},{"assembly_accession":"GCF_000001405.25","chr_name":'
    b'"8","ucsc_style_name":"chr8","refseq_accession":"NC_000008.10","role":"assembled-mol'
    b'ecule","sequence_name":"8","length":300000000},{"assembly_accession":"GCF_000001405.'
    b'25","chr_name":"9","ucsc_style_name":"chr9","refseq_accession":"NC_000009.11","role"'
    b':"assembled-molecule","sequence_name":"9","length":300000000},{"assembly_accession":'
    b'"GCF_000001405.25","chr_name":"10","ucsc_style_name":"chr10","refseq_accession":"NC_'
    b'000010.10","role":"assembled-molecule","sequence_name":"10","length":300000000},{"as'
    b'sembly_accession":"GCF_000001405.25","chr_name":"11","ucsc_style_name":"chr11","refs'
    b'eq_accession":"NC_000011.9","role":"assembled-molecule","sequence_name":"11","length'
    b'":300000000},{"assembly_accession":"GCF_000001405.25","chr_name":"12","ucsc_style_na'
    b'me":"chr12","refseq_accession":"NC_000012.11","role":"assembled-molecule","sequence_'
    b'name":"12","length":300000000},{"assembly_accession":"GCF_000001405.25","chr_name":"'
    b'13","ucsc_style_name":"chr13","refseq_accession":"NC_000013.10","role":"assembled-mo'
    b'lecule","sequence_name":"13","length":300000000},{"assembly_accession":"GCF_00000140'
    b'5.25","chr_name":"14","ucsc_style_name":"chr14","refseq_accession":"NC_000014.8","ro'
    b'le":"assembled-molecule","sequence_name":"14","length":300000000},{"assembly_accessi'
    b'on":"GCF_000001405.25","chr_name":"15","ucsc_style_name":"chr15","refseq_accession":'
    b'"NC_000015.9","role":"assembled-molecule","sequence_name":"15","length":300000000},{'
    b'"assembly_accession":"GCF_000001405.25","chr_name":"16","ucsc_style_name":"chr16","r'
    b'efseq_accession":"NC_000016.9","role":"assembled-molecule","sequence_name":"16","len'
    b'gth":300000000},{"assembly_accession":"GCF_000001405.25","chr_name":"17","ucsc_style'
    b'_name":"chr17","refseq_accession":"NC_000017.10","role":"assembled-molecule","sequen'
    b'ce_name":"17","length":300000000},{"assembly_accession":"GCF_000001405.25","chr_name'
    b'":"18","ucsc_style_name":"chr18","refseq_accession":"NC_000018.9","role":"assembled-'
    b'molecule","sequence_name":"18","length":300000000},{"assembly_accession":"GCF_000001'
    b'405.25","chr_name":"19","ucsc_style_name":"chr19","refseq_accession":"NC_000019.9","'
    b'role":"assembled-molecule","sequence_name":"19","length":300000000},{"assembly_acces'
    b'sion":"GCF_000001405.25","chr_name":"20","ucsc_style_name":"chr20","refseq_accession'
    b'":"NC_000020.10","role":"assembled-molecule","sequence_name":"20","length":300000000'
    b'},{"assembly_accession":"GCF_000001405.25","chr_name":"21","ucsc_style_name":"chr21"'
    b',"refseq_accession":"NC_000021.8","role":"assembled-molecule","sequence_name":"21","'
    b'length":300000000},{"assembly_accession":"GCF_000001405.25","chr_name":"22","ucsc_st'
    b'yle_name":"chr22","refseq_accession":"NC_000022.10","role":"assembled-molecule","seq'
    b'uence_name":"22","length":300000000},{"assembly_accession":"GCF_000001405.25","chr_n'
    b'ame":"X","ucsc_style_name":"chrX","refseq_accession":"NC_000023.10","role":"assemble'
    b'd-molecule","sequence_name":"X","length":300000000},{"assembly_accession":"GCF_00000'
    b'1405.25","chr_name":"Y","ucsc_style_name":"chrY","refseq_accession":"NC_000024.9","r'
    b'ole":"assembled-molecule","sequence_name":"Y","length":300000000},{"assembly_accessi'
    b'on":"GCF_000001405.25","chr_name":"MT","ucsc_style_name":"chrM","refseq_accession":"'
    b'NC_012920.1","role":"assembled-molecule","sequence_name":"MT","length":16569}],"tota'
    b'l_count":25}'
)

# A recorded live reply: NCBI's plain report of Gallus_gallus-5.0, a replaced and
# suppressed assembly, is exactly ``{}``; its any-version report (3,672 B) is kept
# here byte for byte.
GALGAL5_PLAIN_REPORT = b"{}"
GALGAL5_ANY_VERSION_REPORT = (
    b'{"reports":[{"accession":"GCF_000002315.4","current_accession":"GCF_000002315.4"'
    b',"paired_accession":"GCA_000002315.3","source_database":"SOURCE_DATABASE_REFSEQ"'
    b',"organism":{"tax_id":9031,"organism_name":"Gallus gallus","common_name":"chicke'
    b'n","infraspecific_names":{"breed":"Red Jungle fowl, inbred line UCD001","isolate'
    b'":"RJF #256","sex":"female"}},"assembly_info":{"assembly_level":"Chromosome","as'
    b'sembly_status":"suppressed","paired_assembly":{"accession":"GCA_000002315.3","st'
    b'atus":"previous","only_refseq":"chromosome MT","refseq_genbank_are_different":tr'
    b'ue,"differences":"Added chromosome MT"},"assembly_name":"Gallus_gallus-5.0","ass'
    b'embly_type":"haploid","bioproject_lineage":[{"bioprojects":[{"accession":"PRJNA1'
    b'3342","title":"The Chicken Genome Project, currently maintained by the Genome Re'
    b'ference Consortium (GRC)","parent_accessions":["PRJNA167910"]},{"accession":"PRJ'
    b'NA167910","title":"NHGRI Large Scale Sequencing Program for Organismal Sequencin'
    b'g"}]}],"bioproject_accession":"PRJNA13342","release_date":"2015-12-16","submitte'
    b'r":"International Chicken Genome Consortium","sequencing_tech":"Sanger; 454; Ill'
    b'umina; PacBio","assembly_method":"MHAP/PBcR v. 8.2beta","biosample":{"accession"'
    b':"SAMN02981218","last_updated":"2015-12-16T07:55:36.780","publication_date":"201'
    b'4-08-11T11:15:50.240","submission_date":"2014-08-11T11:15:50.240","sample_ids":['
    b'{"db":"SRA","value":"SRS926532"},{"db":"GenBank","value":"gb|AADN00000000.3"}],"'
    b'description":{"title":"Sample from Gallus gallus","organism":{"tax_id":9031,"org'
    b'anism_name":"Gallus gallus"}},"owner":{"name":"The Genome Institute, Washington '
    b'University School of Medicine"},"models":["Generic"],"bioprojects":[{"accession"'
    b':"PRJNA13342"}],"package":"Generic.1.0","attributes":[{"name":"sex","value":"fem'
    b'ale"},{"name":"isolate","value":"RJF #256"},{"name":"breed","value":"Red Jungle '
    b'fowl, inbred line UCD001"}],"status":{"status":"live","when":"2014-08-11T11:15:5'
    b'0.240"},"breed":"Red Jungle fowl, inbred line UCD001","isolate":"RJF #256","sex"'
    b':"female"},"blast_url":"https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=Blast'
    b'Search&PROG_DEF=blastn&BLAST_SPEC=GDH_GCF_000002315.4","comments":"Assembly name'
    b": Gallus_gallus-5.0\\nThe genomic sequence for this RefSeq record is from the who"
    b"le-genome assembly released by the International Chicken Genome Consortium on 20"
    b"15/12/16. The original whole-genome shotgun project has the accession AADN000000"
    b'00.4."},"assembly_stats":{"total_number_of_chromosomes":35,"total_sequence_lengt'
    b'h":"1230241782","total_ungapped_length":"1218476783","number_of_contigs":24692,"'
    b'contig_n50":2894815,"contig_l50":110,"number_of_scaffolds":23869,"scaffold_n50":'
    b'6379610,"scaffold_l50":47,"gaps_between_scaffolds_count":397,"number_of_componen'
    b't_sequences":25474,"atgc_count":"1285610588","gc_count":"549915393","gc_percent"'
    b':43,"genome_coverage":"70","number_of_organelles":1},"organelle_info":[{"descrip'
    b'tion":"Mitochondrion","total_seq_length":"16775","submitter":"Genome Reference C'
    b'onsortium"}],"additional_submitters":[{"genbank_accession":"X52392.1","refseq_ac'
    b'cession":"NC_001323.1","chr_name":"MT","molecule_type":"Mitochondrion","submitte'
    b'r":"Morais R., Departement de Biochemie, Universite de Montreal, C.P. 6128, Succ'
    b'. A, Montreal (Quebec), H3C #j7, Canada"}],"annotation_info":{"name":"Annotation'
    b' submitted by NCBI RefSeq","provider":"NCBI RefSeq","release_date":"2016-01-04",'
    b'"stats":{"gene_counts":{"total":26343,"protein_coding":19073,"non_coding":6550,"'
    b'pseudogene":627,"other":93}}},"wgs_info":{"wgs_project_accession":"AADN04","mast'
    b'er_wgs_url":"https://www.ncbi.nlm.nih.gov/nuccore/AADN00000000.4","wgs_contigs_u'
    b'rl":"https://www.ncbi.nlm.nih.gov/Traces/wgs/AADN04"}}],"total_count":1}'
)


# The same rows as the FTP assembly report `ncbi_sequence_names` answers, derived row
# for row by `tests.sequence_naming_support.assembly_report_text`: no GET.
from tests.sequence_naming_support import assembly_report_text  # noqa: E402

GRCH38_NAMES = assembly_report_text(GRCH38_MOLECULES, header="GRCh38.p14")
GRCH37_NAMES = assembly_report_text(GRCH37_MOLECULES, header="GRCh37.p13")
SSCROFA11_NAMES = assembly_report_text(SSCROFA11_MOLECULES, header="Sscrofa11.1")
