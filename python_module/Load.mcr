#!MC 1410
$!Page Name = 'Untitled'
$!PageControl Create
$!NewLayout
$!FrameLayout ShowBorder = No
$!PrintSetup Palette = Color
$!ExportSetup UseSuperSampleAntiAliasing = Yes
$!ExportSetup ImageWidth = 2000

#-------------------------------------------------
# Turbulence Model Variants
#------------------------------
$!VarSet |SA_Maps|         = ''
$!VarSet |SAQ_Maps|        = ''
$!VarSet |SST_Maps|        = ''
$!VarSet |WMLES_Maps|      = ''
#-------------------------------------------------
# Grid Type Variants
#------------------------------
$!VarSet |CadenceUnSTet_Maps| = ''
$!VarSet |CadenceUnSVox_Maps| = ''
$!VarSet |HeldenUnSt_Maps| = ''
$!VarSet |CstmUsrAdp_Maps| = ''
$!VarSet |CstmUsrUns_Maps| = ''
$!VarSet |CstmUsrStr_Maps| = ''
#-------------------------------------------------
# Wing Type Variants
#------------------------------
$!VarSet |WING1_Maps| = ''
$!VarSet |WING2_Maps| = ''
#-------------------------------------------------
# DPW3 Variants
#------------------------------
$!VarSet |DPW3_Maps| = ''
#-------------------------------------------------

#===================================================================
#$!RUNMACROFUNCTION "LoadAndPlotGrid" ( '|GRID|', '|IMG1|', '|IMG2|' )
##################################################
#$!MACROFUNCTION
#  NAME = "LoadAndPlotGrid"
#  $!NewLayout
#  $!VarSet |GRID|  = '|1|'
#  $!VarSet |ALPHA| =  '0'
#  $!VarSet |IMG1|  = '|2|'
#  $!VarSet |IMG2|  = '|3|'
#  $!IF '|GRID|' == 'N'
#    $!VarSet |Increment| = 6
#  $!ELSE
#    $!VarSet |Increment| = |GRID|
#  $!ENDIF
#  $!RUNMACROFUNCTION "LoadData"
#  $!INCLUDEMACRO "|MACROFILEPATH|/Plot_Grid.mcr"
#$!ENDMACROFUNCTION
#===================================================================
#$!RUNMACROFUNCTION "LoadAndPlotAlpha" ( '|ALPHA|', '|IMG1|', '|IMG2|' )
##################################################
$!MACROFUNCTION
  NAME = "LoadAndPlotAlpha"
  $!NewLayout
  $!VarSet |GRID|  =  '0'
  $!VarSet |ALPHA| = '|1|'
  $!VarSet |IMG1|  = '|2|'
  $!VarSet |IMG2|  = '|3|'
  $!IF     |ALPHA| == 0.50
    $!VarSet |Increment| = 8
  $!ELSEIF |ALPHA| == 1.50
    $!VarSet |Increment| = 8
  $!ENDIF
  $!RUNMACROFUNCTION "LoadData"
  $!INCLUDEMACRO "|MACROFILEPATH|/Plot_Alpha.mcr"
$!ENDMACROFUNCTION
#===================================================================
#$!RUNMACROFUNCTION "AppendDataSetFile"
##################################################
$!MACROFUNCTION
  NAME = "AppendDataSetFile"
  $!ReadDataSet  '|1|'
    ReadDataOption = Append
    ResetStyle = No
    VarLoadMode = ByName
    AssignStrandIDs = Yes
$!ENDMACROFUNCTION
#===================================================================
#$!RUNMACROFUNCTION "CalculateDeltas" ( 8 16 )
##################################################
$!MACROFUNCTION
  NAME = "CalculateDeltas"
  $!AlterData  [|1|]
    IgnoreDivideByZero = Yes
    Equation = '{<greek>D</greek>CL<sub>Total</sub>}        = ({CL<sub>Total</sub>}[|1|]        - {CL<sub>Total</sub>}[|2|])'
  $!AlterData  [|1|]
    IgnoreDivideByZero = Yes
    Equation = '{<greek>D</greek>CMy<sub>Total</sub>}       = ({CMy<sub>Total</sub>}[|1|]       - {CMy<sub>Total</sub>}[|2|])'
  $!AlterData  [|1|]
    IgnoreDivideByZero = Yes
    Equation = '{<greek>D</greek>CD<sub>Total</sub>}        = ({CD<sub>Total</sub>}[|1|]        - {CD<sub>Total</sub>}[|2|])        * 10000.0'
  $!AlterData  [|1|]
    IgnoreDivideByZero = Yes
    Equation = '{<greek>D</greek>CD<sub>Pressure</sub>}     = ({CD<sub>Pressure</sub>}[|1|]     - {CD<sub>Pressure</sub>}[|2|])     * 10000.0'
  $!AlterData  [|1|]
    IgnoreDivideByZero = Yes
    Equation = '{<greek>D</greek>CD<sub>SkinFriction</sub>} = ({CD<sub>SkinFriction</sub>}[|1|] - {CD<sub>SkinFriction</sub>}[|2|]) * 10000.0'
$!ENDMACROFUNCTION
#===================================================================
#$!RUNMACROFUNCTION "AddLineMap"
##################################################
$!MACROFUNCTION
  NAME = "AddLineMap"
  $!IF |Z| > |NumZones|
    $!VarSet |Z| = |NumZones|
  $!ENDIF
  $!CreateLineMap
  $!LineMap [|NumLineMaps|]  Name = '&(ZONENAME[|Z|]%6.6s)'
  $!LineMap [|NumLineMaps|]  Assign{Zone = |Z|}
  $!ActiveLineMaps += [|NumLineMaps|]
  # Assign LineMap to Turbulence Model Label
  $!IF     '|1|' == 'SA_Maps'
     $!VarSet |SA_Maps|         = '|SA_Maps| |NumLineMaps|,'
  $!ELSEIF '|1|' == 'SAQ_Maps'
     $!VarSet |SAQ_Maps|        = '|SAQ_Maps| |NumLineMaps|,'
  $!ELSEIF '|1|' == 'SST_Maps'
     $!VarSet |SST_Maps|        = '|SST_Maps| |NumLineMaps|,'
  $!ELSEIF '|1|' == 'WMLES_Maps'
     $!VarSet |WMLES_Maps|      = '|WMLES_Maps| |NumLineMaps|,'
  $!ENDIF
  # Assign LineMap to Grid Type Label
  $!IF     '|2|' == 'CadenceUnSTet_Maps'
     $!VarSet |CadenceUnSTet_Maps| = '|CadenceUnSTet_Maps| |NumLineMaps|,'
  $!ELSEIF '|2|' == 'CadenceUnSVox_Maps'
     $!VarSet |CadenceUnSVox_Maps| = '|CadenceUnSVox_Maps| |NumLineMaps|,'
  $!ELSEIF '|2|' == 'HeldenUnSt_Maps'
     $!VarSet |HeldenUnSt_Maps|    = '|HeldenUnSt_Maps|    |NumLineMaps|,'
  $!ELSEIF '|2|' == 'CstmUsrAdp_Maps'
     $!VarSet |CstmUsrAdp_Maps|    = '|CstmUsrAdp_Maps|    |NumLineMaps|,'
  $!ELSEIF '|2|' == 'CstmUsrUns_Maps'
     $!VarSet |CstmUsrUns_Maps|    = '|CstmUsrUns_Maps|    |NumLineMaps|,'
  $!ELSEIF '|2|' == 'CstmUsrAdp_Maps'
     $!VarSet |CstmUsrStr_Maps|    = '|CstmUsrStr_Maps|    |NumLineMaps|,'
  $!ELSEIF '|2|' == 'DPW3_Maps'
     $!VarSet |DPW3_Maps|          = '|DPW3_Maps|          |NumLineMaps|,'
  $!ENDIF
  # Assign LineMap to Wing Type Label
  $!IF     '|3|' == 'WING1'
    $!VarSet |WING1_Maps| = '|WING1_Maps| |NumLineMaps|,'
  $!ELSEIF '|3|' == 'WING2'
    $!VarSet |WING2_Maps| = '|WING2_Maps| |NumLineMaps|,'
  $!ENDIF
$!ENDMACROFUNCTION
#===================================================================
#$!RUNMACROFUNCTION "ExportAndSave" ( '|MACROFILEPATH|',  'Grid|GRID|_|PlotName|',  '|ImageFormat|', '|ImageFormat2|' )
#$!RUNMACROFUNCTION "ExportAndSave" ( '|MACROFILEPATH|', 'Alpha|ALPHA|_|PlotName|', '|ImageFormat|', '|ImageFormat2|' )
##################################################
$!MACROFUNCTION
  NAME = "ExportAndSave"
  $!VarSet |PATH| = '|1|'
  $!VarSet |FILE| = '|2|'
  $!VarSet |IMGF| = '|3|'
  $!VarSet |IMG2| = '|4|'
  $!IF     '|IMGF|' == 'eps'  # do nothing ... supported
  $!ELSEIF '|IMGF|' == 'png'  # do nothing ... supported
  $!ELSEIF '|IMGF|' == 'bmp'  # do nothing ... supported
  $!ELSEIF '|IMGF|' == 'jpeg' # do nothing ... supported
  $!ELSEIF '|IMGF|' == 'ps'   # do nothing ... supported
  $!ELSEIF '|IMGF|' == 'tiff' # do nothing ... supported
  $!ELSE
     $!VarSet |IMGF| = ''     # disabled ... unsupported
  $!ENDIF
  $!IF     '|IMG2|' == 'eps'  # do nothing ... supported
  $!ELSEIF '|IMG2|' == 'png'  # do nothing ... supported
  $!ELSEIF '|IMG2|' == 'bmp'  # do nothing ... supported
  $!ELSEIF '|IMG2|' == 'jpeg' # do nothing ... supported
  $!ELSEIF '|IMG2|' == 'ps'   # do nothing ... supported
  $!ELSEIF '|IMG2|' == 'tiff' # do nothing ... supported
  $!ELSE
     $!VarSet |IMG2| = ''     # disabled ... unsupported
  $!ENDIF
  $!RedrawAll
  $!IF '|IMG2|' != ''
    $!ExportSetup ExportFormat = |IMG2|
    $!ExportSetup ExportFName = '|PATH|/|IMG2|/|FILE|.|IMG2|'
   # Linux Command: Create ouput directory (not general)
   #$!SYSTEM 'mkdir -p |PATH|/|IMG2|'
    $!Export 
      ExportRegion = AllFrames
  $!ENDIF
  $!IF '|IMGF|' != ''
    $!ExportSetup ExportFormat = |IMGF|
    $!ExportSetup ExportFName = '|PATH|/|IMGF|/|FILE|.|IMGF|'
   # Linux Command: Create ouput directory (not general)
   #$!SYSTEM 'mkdir -p |PATH|/|IMGF|'
    $!Export
      ExportRegion = AllFrames
  $!ENDIF
 # Linux Command: Create ouput directory (not general)
 #$!SYSTEM 'mkdir -p |PATH|/lay'
  $!SaveLayout  '|PATH|/lay/|FILE|.lay'
    UseRelativePaths = Yes
  $!ExportSetup ExportFormat = |IMGF|
$!ENDMACROFUNCTION
#===================================================================
#$!RUNMACROFUNCTION "LoadData"
##################################################
$!MACROFUNCTION
  NAME = "LoadData"
#-------------------------------------------------
# Map00:
#$!VarSet |Z| = ( 0 + |Increment| )
#$!ReadDataSet  '"|MACROFILEPATH|/000_00_DPW8-AePW4_ForceMoment_v5.dat"'
#==================================================================================================
# Dataset: 301.01
$!ReadDataSet  '"|MACROFILEPATH|/DPW3/301_FUN3D_Lee-Rausch_Rumsey/01_SA/DPW8-AePW4_ForceMoment_v6.dat"'
  ReadDataOption = New
  ResetStyle = No
  VarLoadMode = ByName
  AssignStrandIDs = Yes
  VarNameList = '"GRID_LEVEL" "GRID_SIZE" "GRID_FAC" "MACH" "REY" "ALPHA" "CL_TOT" "CD_TOT" "CM_TOT" "CL_WING" "CD_WING" "CM_WING" "CD_PR" "CD_SF" "CL_TAIL" "CD_TAIL" "CM_TAIL" "CL_FUS" "CD_FUS" "CM_FUS" "CL_NAC" "CD_NAC" "CM_NAC" "CL_PY" "CD_PY" "CM_PY" "CPU_Hours" "DELTAT" "CTUSTART" "CTUAVG" "Q/E"'
$!ActiveLineMaps -= [1-|NumLineMaps|]
$!DeleteLineMaps    [1-|NumLineMaps|]
$!VarSet |Z| = (  0  + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#
$!VarSet |N0| = ( |NumLineMaps| - 1 )
$!LineMap [|N0|-|NumLineMaps|]  Lines   { Color = Red    } Symbols {Color = Red    FillColor = Red    SymbolShape {IsASCII = Yes ASCIIShape {FontOverride = UserDef ASCIIChar = '\1'}}}
#-------------------------------------------------
# Dataset: 302.01
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/DPW3/302_NSU3D_Mavriplis/01_SA/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#
$!VarSet |N0| = ( |NumLineMaps| - 1 )
$!LineMap [|N0|-|NumLineMaps|]  Lines   { Color = Green  } Symbols {Color = Green  FillColor = Green  SymbolShape {IsASCII = Yes ASCIIShape {FontOverride = UserDef ASCIIChar = '\2'}}}
#-------------------------------------------------
# Dataset: 303.01
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/DPW3/303_OVERFLOW_Sclafani/01_SA/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#
$!VarSet |N0| = ( |NumLineMaps| - 1 )
$!LineMap [|N0|-|NumLineMaps|]  Lines   { Color = Blue   } Symbols {Color = Blue   FillColor = Blue   SymbolShape {IsASCII = Yes ASCIIShape {FontOverride = UserDef ASCIIChar = '\3'}}}
#-------------------------------------------------
# Dataset: 305.01
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/DPW3/305_NSU3D_Zickuhr/01_SA/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#
$!VarSet |N0| = ( |NumLineMaps| - 1 )
$!LineMap [|N0|-|NumLineMaps|]  Lines   { Color = Purple } Symbols {Color = Purple FillColor = Purple SymbolShape {IsASCII = Yes ASCIIShape {FontOverride = UserDef ASCIIChar = '\5'}}}
#-------------------------------------------------
# Dataset: 306.01
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/DPW3/306_CFL3D_Tinoco/01_SA/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#
$!VarSet |N0| = ( |NumLineMaps| - 1 )
$!LineMap [|N0|-|NumLineMaps|]  Lines   { Color = Custom3  } Symbols {Color = Custom3  FillColor = Custom3  SymbolShape {IsASCII = Yes ASCIIShape {FontOverride = UserDef ASCIIChar = '\6'}}}
#-------------------------------------------------
# Dataset: 306.02
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/DPW3/306_CFL3D_Tinoco/02_SST/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#
$!VarSet |N0| = ( |NumLineMaps| - 1 )
$!LineMap [|N0|-|NumLineMaps|]  Lines   { Color = Custom32 } Symbols {Color = Custom32 FillColor = Custom32 SymbolShape {IsASCII = Yes ASCIIShape {FontOverride = UserDef ASCIIChar = '\7'}}}
#-------------------------------------------------
# Dataset: 307.01
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/DPW3/307_TAU_Brodersen/01_SA/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#
$!VarSet |N0| = ( |NumLineMaps| - 1 )
$!LineMap [|N0|-|NumLineMaps|]  Lines   { Color = Custom29 } Symbols {Color = Custom29 FillColor = Custom29 SymbolShape {IsASCII = Yes ASCIIShape {FontOverride = UserDef ASCIIChar = '\8'}}}
#-------------------------------------------------
# Dataset: 308.01
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/DPW3/308_FLUENT_Scheidegger/01_KE/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "DPW3_Maps" "DPW3_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#
$!VarSet |N0| = ( |NumLineMaps| - 1 )
$!LineMap [|N0|-|NumLineMaps|]  Lines   { Color = Custom15 } Symbols {Color = Custom15 FillColor = Custom15 SymbolShape {IsASCII = Yes ASCIIShape {FontOverride = UserDef ASCIIChar = '\9'}}}
#==================================================================================================
# Map01: 003.01
$!VarSet |Z| = ( |NumZones| + |Increment| - 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/003_Boeing/01_FELight_HeldenMesh_Unstructured_REV00_SA-neg/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "HeldenUnSt_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 8 )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "HeldenUnSt_Maps" "WING2")
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#-------------------------------------------------
# Map02: 003.02
$!VarSet |Z| = ( |NumZones| + |Increment| - 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/003_Boeing/02_FELight_Cadence_Unstructured_HexTrex_TetFF_REV00_SA-neg/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CadenceUnSTet_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + |Increment| )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CadenceUnSTet_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#-------------------------------------------------
# Map03: 003.03
$!VarSet |Z| = ( |NumZones| + |Increment| - 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/003_Boeing/03_FELight_Cadence_Unstructured_HexTrex_VoxFF_REV00_SA-neg/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CadenceUnSVox_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + |Increment| )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CadenceUnSVox_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#-------------------------------------------------
$!LineMap [17]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\A'}}}
$!LineMap [18]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\A'}}}
$!LineMap [19]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\a'}}}
$!LineMap [20]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\a'}}}
$!LineMap [21]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {UseBaseFont = No FontOverride = Greek   ASCIIChar = '\A'}}}
$!LineMap [22]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {UseBaseFont = No FontOverride = Greek   ASCIIChar = '\A'}}}
#-------------------------------------------------
# Map04: 003.04
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/003_Boeing/04_GGNST1_Epic_DragAdjoint_SA-neg/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CstmUsrAdp_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CstmUsrAdp_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#-------------------------------------------------
# Map05: 003.05
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/003_Boeing/05_GGNST1_HeldenMesh_Unstructured_REV00_SA-neg/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "HeldenUnSt_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "HeldenUnSt_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#-------------------------------------------------
# Map06: 003.06
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/003_Boeing/06_GGNST1_Cadence_Unstructured_HexTrex_TetFF_REV00_SA-neg/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CadenceUnSTet_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CadenceUnSTet_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#-------------------------------------------------
# Map07: 003.07
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/003_Boeing/07_GGNST1_Cadence_Unstructured_HexTrex_VoxFF_REV00_SA-neg/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CadenceUnSVox_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CadenceUnSVox_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#-------------------------------------------------
$!LineMap [23]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\B'}}}
$!LineMap [24]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\B'}}}
$!LineMap [25]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\b'}}}
$!LineMap [26]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\b'}}}
$!LineMap [27]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {UseBaseFont = No FontOverride = Greek   ASCIIChar = '\B'}}}
$!LineMap [28]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {UseBaseFont = No FontOverride = Greek   ASCIIChar = '\B'}}}
$!LineMap [29]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {UseBaseFont = No FontOverride = Greek   ASCIIChar = '\b'}}}
$!LineMap [30]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {UseBaseFont = No FontOverride = Greek   ASCIIChar = '\b'}}}
#-------------------------------------------------
# Map08: 011.01
$!VarSet |Z1| = (|NumZones|+1)
$!VarSet |Z| = ( |NumZones| + 12 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/011_HEMLAB/01_Adaptive/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CstmUsrAdp_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 12 )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CstmUsrAdp_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
  #-------------------------------------------------
  # CUSTOM: (KLUDGE) CM = -1 * CM because it appears to have the wrong sign
  $!AlterData  [|Z1|-|NumZones|]
    IgnoreDivideByZero = Yes
    Equation = 'V9 = -1 * V9'
#-------------------------------------------------
# Map09: 037.01
$!VarSet |Z| = ( |NumZones| + |Increment| - 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/037_HeldenAerospace/01_USM3DME_HeldenMesh_Unstructured_REV00_SA-neg/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "HeldenUnSt_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + |Increment| )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "HeldenUnSt_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#-------------------------------------------------
# Map10: 003.08
$!VarSet |Z| = ( |NumZones| + 1 )
$!RUNMACROFUNCTION "AppendDataSetFile" ( '"|MACROFILEPATH|/../../../../DPW8-Scatter/TestCase3a/003_Boeing/08_GGNST1_Epic_DragAdjoint_SA-neg/DPW8-AePW4_ForceMoment_v6.dat"' )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CstmUsrAdp_Maps" "WING1" )
$!VarSet |Z| = ( |Z| + 1 )
$!RUNMACROFUNCTION "AddLineMap" ( "SA_Maps" "CstmUsrAdp_Maps" "WING2" )
$!LineMap [|NumLineMaps|]  Assign{ShowInLegend = Never}
#-------------------------------------------------
#==================================================================================================

#==================================================================================================
$!LineMap [31]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\C'}}}
$!LineMap [32]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\C'}}}

$!LineMap [33]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\D'}}}
$!LineMap [34]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\D'}}}

$!LineMap [35]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\E'}}}
$!LineMap [36]  Symbols {SymbolShape {IsASCII = Yes ASCIIShape {                 FontOverride = UserDef ASCIIChar = '\E'}}}
#-------------------------------------------------

#==================================================================================================
# Data Alterations
#-------------------------------------------------
# Establish deltas between Wing1/Wing2
#----------------------------
$!AlterData 
  IgnoreDivideByZero = Yes
  Equation = '{<greek>D</greek>CL<sub>Total</sub>}=0.0'
$!VarSet |dCL| = |NumVars|
$!AlterData 
  IgnoreDivideByZero = Yes
  Equation = '{<greek>D</greek>CMy<sub>Total</sub>}=0.0'
$!VarSet |dCMy| = |NumVars|
$!AlterData 
  IgnoreDivideByZero = Yes
  Equation = '{<greek>D</greek>CD<sub>Total</sub>}=0.0'
$!VarSet |dCD| = |NumVars|
$!AlterData 
  IgnoreDivideByZero = Yes
  Equation = '{<greek>D</greek>CD<sub>Pressure</sub>}=0.0'
$!VarSet |dCDp| = |NumVars|
$!AlterData 
  IgnoreDivideByZero = Yes
  Equation = '{<greek>D</greek>CD<sub>SkinFriction</sub>}=0.0'
$!VarSet |dCDv| = |NumVars|
#-------------------------------------------------
# Calculate Grid Size Factor(s) based on GRID_SIZE variable
#----------------------------
# 1/h
$!AlterData
  IgnoreDivideByZero = Yes
  Equation = '{h = N<sup>-1</sup>} = 1/({GRID_SIZE}**(1/1))'
$!VarSet |1_h| = |NumVars|
#----------------------------
# 1/(h^2)
$!AlterData
  IgnoreDivideByZero = Yes
  Equation = '{h = N<sup>-1/2</sup>} = 1/({GRID_SIZE}**(1/2))'
$!VarSet |1_h1o2| = |NumVars|
#----------------------------
# 1/(h^3)
$!AlterData
  IgnoreDivideByZero = Yes
  Equation = '{h = N<sup>-1/3</sup>} = 1/({GRID_SIZE}**(1/3))'
#  Equation = '{1/h<sup>1/3</sup>} = 1/({GRID_SIZE}**(1/3))'
$!VarSet |1_h1o3| = |NumVars|
#----------------------------
# 1/(h^(2/3))
$!AlterData
  IgnoreDivideByZero = Yes
  Equation = '{h = N<sup>-2/3</sup>} = 1/({GRID_SIZE}**(2/3))'
$!VarSet |1_h2o3| = |NumVars|

#==================================================================================================
# Value blanking for data with AOA <= 0.00
#-------------------------------------------------
#$!Blanking Value { Include = Yes }
#$!Blanking Value { Constraint 1 { Include = Yes ValueCutoff = 0.00 VarA = 6 } }
#$!Blanking Value { Constraint 1 { Include = No } }

#==================================================================================================
# Assign Variable Names
#-------------------------------------------------
$!RenameDataSetVar Var = 6      Name = 'AoA'
$!RenameDataSetVar Var = 7      Name = 'CL<sub>Total</sub>'
$!RenameDataSetVar Var = 8      Name = 'CD<sub>Total</sub>'
$!RenameDataSetVar Var = 9      Name = 'CMy<sub>Total</sub>'
$!RenameDataSetVar Var = 10     Name = 'CL<sub>Wing</sub>'
$!RenameDataSetVar Var = 11     Name = 'CD<sub>Wing</sub>'
$!RenameDataSetVar Var = 12     Name = 'CMy<sub>Wing</sub>'
$!RenameDataSetVar Var = 13     Name = 'CD<sub>Pressure</sub>'
$!RenameDataSetVar Var = 14     Name = 'CD<sub>SkinFriction</sub>'
$!RenameDataSetVar Var = 15     Name = 'CL<sub>Tail</sub>'
$!RenameDataSetVar Var = 16     Name = 'CD<sub>Tail</sub>'
$!RenameDataSetVar Var = 17     Name = 'CMy<sub>Tail</sub>'
$!RenameDataSetVar Var = 18     Name = 'CL<sub>Fuselage</sub>'
$!RenameDataSetVar Var = 19     Name = 'CD<sub>Fuselage</sub>'
$!RenameDataSetVar Var = 20     Name = 'CMy<sub>Fuselage</sub>'
$!RenameDataSetVar Var = 21     Name = 'CL<sub>Nacelle</sub>'
$!RenameDataSetVar Var = 22     Name = 'CD<sub>Nacelle</sub>'
$!RenameDataSetVar Var = 23     Name = 'CMy<sub>Nacelle</sub>'
$!RenameDataSetVar Var = 24     Name = 'CL<sub>Pylon</sub>'
$!RenameDataSetVar Var = 25     Name = 'CD<sub>Pylon</sub>'
$!RenameDataSetVar Var = 26     Name = 'CMy<sub>Pylon</sub>'
$!RenameDataSetVar Var = 27     Name = 'CPU<sub>Hours</sub>'
$!RenameDataSetVar Var = 28     Name = '<greek>D</greek>T'
$!RenameDataSetVar Var = 29     Name = 'CTU<sub>Start</sub>'
$!RenameDataSetVar Var = 30     Name = 'Q/E'
$!RenameDataSetVar Var = |dCL|  Name = '<greek>D</greek>CL<sub>Total</sub>'
$!RenameDataSetVar Var = |dCMy| Name = '<greek>D</greek>CMy<sub>Total</sub>'
$!RenameDataSetVar Var = |dCD|  Name = '<greek>D</greek>CD<sub>Total</sub>'
$!RenameDataSetVar Var = |dCDp| Name = '<greek>D</greek>CD<sub>Pressure</sub>'
$!RenameDataSetVar Var = |dCDv| Name = '<greek>D</greek>CD<sub>SkinFriction</sub>'

#-------------------------------------------------
# Calculate deltas between Wing1/Wing2 (HARDCODED)
#----------------------------
$!RUNMACROFUNCTION "CalculateDeltas" ( 24, 32 )
$!RUNMACROFUNCTION "CalculateDeltas" ( 39, 46 )
$!RUNMACROFUNCTION "CalculateDeltas" ( 53, 60 )
$!RUNMACROFUNCTION "CalculateDeltas" ( 61, 62 )
$!RUNMACROFUNCTION "CalculateDeltas" ( 63, 64 )
$!RUNMACROFUNCTION "CalculateDeltas" ( 65, 66 )
$!RUNMACROFUNCTION "CalculateDeltas" ( 67, 68 )
$!RUNMACROFUNCTION "CalculateDeltas" ( 80, 91 )
$!RUNMACROFUNCTION "CalculateDeltas" ( 98,105 )
$!RenameDataSetVar Var = |dCD|  Name = '<greek>D</greek>CD<sub>Total (cnts)</sub>'
$!RenameDataSetVar Var = |dCDp| Name = '<greek>D</greek>CD<sub>Pressure (cnts)</sub>'
$!RenameDataSetVar Var = |dCDv| Name = '<greek>D</greek>CD<sub>SkinFriction (cnts)</sub>'

#==================================================================================================
$!LineMap [1-|NumLineMaps|]  Symbols { Size = 1.0 }
$!XYLineAxis XDetail 1 { CoordScale = Linear Gridlines { Show = Yes } MinorGridlines { Show = Yes } Title { Offset =  6 } }
$!XYLineAxis YDetail 1 { CoordScale = Linear Gridlines { Show = Yes } MinorGridlines { Show = Yes } Title { Offset = 10 } }
$!XYLineAxis GridArea  { DrawBorder = Yes LineThickness = 0.1 }

#-------------------------------------------------
$!LinePlotLayers
  ShowSymbols = Yes

$!GlobalLinePlot
  DataLabels { DistanceSkip = 5 }
  Legend {
    Show = Yes
    TextShape { Height = 1.2 }
    Box { BoxType = None Margin = 5 }
    RowSpacing = 1.2
    XYPos { X = 88 Y = 10 }
    AnchorAlignment = BottomLeft
    }

$!ENDMACROFUNCTION
#===================================================================
